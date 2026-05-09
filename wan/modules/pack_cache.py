from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F


@dataclass
class PackCacheConfig:
    """Configuration for the block-level PackForcing cache backend."""

    frame_seq_len: int
    num_frame_per_block: int
    sink_blocks: int
    recent_blocks: int
    mid_bank_capacity_blocks: int
    mid_select_topk_blocks: int
    compress_mode: str = "token_avg_pool"
    evict_mode: str = "fifo"
    reuse_mid_selection_within_block: bool = True
    enable_rope_adjustment: bool = False

    def __post_init__(self) -> None:
        int_fields = [
            "frame_seq_len",
            "num_frame_per_block",
            "sink_blocks",
            "recent_blocks",
            "mid_bank_capacity_blocks",
            "mid_select_topk_blocks",
        ]
        for name in int_fields:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.evict_mode not in {"fifo"}:
            raise ValueError(f"Unsupported evict_mode={self.evict_mode}.")


@dataclass(frozen=True)
class BlockMeta:
    """Metadata describing one full-res or compressed historical block."""

    block_id: int
    start_frame: int
    num_frames: int
    packed_frame_span: int
    tokens_per_frame: int
    num_tokens: int
    is_compressed: bool
    source_block_id: int | None = None

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if self.packed_frame_span <= 0:
            raise ValueError(f"packed_frame_span must be positive, got {self.packed_frame_span}.")
        if self.tokens_per_frame <= 0:
            raise ValueError(f"tokens_per_frame must be positive, got {self.tokens_per_frame}.")
        if self.num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {self.num_tokens}.")

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.num_frames


@dataclass
class FullResKVBlock:
    k: torch.Tensor  # [B, N, H, D]
    v: torch.Tensor  # [B, N, H, D]
    meta: BlockMeta
    precomputed_mid: CompressedKVBlock | None = None


@dataclass
class CompressedKVBlock:
    k: torch.Tensor  # [B, Nc, H, D]
    v: torch.Tensor  # [B, Nc, H, D]
    meta: BlockMeta


@dataclass
class SourceLatentBlock:
    latent: torch.Tensor  # [B, C, T, H, W]
    meta: BlockMeta


@dataclass
class PackSourceCacheState:
    """Shared latent-space history used for train-time mid recomputation."""

    recent_blocks: deque[SourceLatentBlock] = field(default_factory=deque)
    mid_blocks: deque[SourceLatentBlock] = field(default_factory=deque)
    next_block_id: int = 0
    total_committed_blocks: int = 0
    batch_size: int | None = None
    last_block_start_frame: int | None = None


@dataclass
class PackAttentionView:
    """History tensors and metadata for one attention call."""

    history_k: torch.Tensor  # [B, N_hist, H, D]
    history_v: torch.Tensor  # [B, N_hist, H, D]
    sink_token_count: int
    mid_token_count: int
    recent_token_count: int
    ordered_block_meta_per_batch: list[list[BlockMeta]]
    packed_start_frames_per_batch: list[list[int]]
    expected_query_start_frames: list[int]  # Anchor end frame used to backward-pack history.


@dataclass
class PackLayerCacheState:
    """Per-transformer-layer PackForcing cache state."""

    cfg: PackCacheConfig
    sink_blocks: list[FullResKVBlock] = field(default_factory=list)
    recent_blocks: deque[FullResKVBlock] = field(default_factory=deque)
    mid_blocks: deque[CompressedKVBlock] = field(default_factory=deque)
    next_block_id: int = 0
    total_committed_blocks: int = 0
    batch_size: int | None = None
    cached_selected_mid_indices: torch.Tensor | None = None
    cached_selection_for_start_frame: int | None = None


class PackBlockCompressor(Protocol):
    def __call__(self, block: FullResKVBlock) -> CompressedKVBlock:
        ...


class TokenAverageCompressor:
    """A placeholder block compressor that average-pools along the token axis."""

    def __init__(self, target_tokens_per_block: int) -> None:
        if target_tokens_per_block <= 0:
            raise ValueError("target_tokens_per_block must be positive.")
        self.target_tokens_per_block = int(target_tokens_per_block)

    def __call__(self, block: FullResKVBlock) -> CompressedKVBlock:
        k = _adaptive_avg_pool_tokens(block.k, self.target_tokens_per_block)
        v = _adaptive_avg_pool_tokens(block.v, self.target_tokens_per_block)
        meta = BlockMeta(
            block_id=block.meta.block_id,
            start_frame=block.meta.start_frame,
            num_frames=block.meta.num_frames,
            packed_frame_span=block.meta.packed_frame_span,
            tokens_per_frame=block.meta.tokens_per_frame,
            num_tokens=k.shape[1],
            is_compressed=True,
            source_block_id=block.meta.block_id,
        )
        return CompressedKVBlock(k=k, v=v, meta=meta)


class IdentityCompressor:
    """A no-op compressor useful for bootstrapping the state machine."""

    def __call__(self, block: FullResKVBlock) -> CompressedKVBlock:
        meta = BlockMeta(
            block_id=block.meta.block_id,
            start_frame=block.meta.start_frame,
            num_frames=block.meta.num_frames,
            packed_frame_span=block.meta.packed_frame_span,
            tokens_per_frame=block.meta.tokens_per_frame,
            num_tokens=block.meta.num_tokens,
            is_compressed=True,
            source_block_id=block.meta.block_id,
        )
        return CompressedKVBlock(k=block.k.clone(), v=block.v.clone(), meta=meta)


def init_empty_pack_layer_cache(cfg: PackCacheConfig) -> PackLayerCacheState:
    return PackLayerCacheState(cfg=cfg)


def reset_pack_layer_cache(state: PackLayerCacheState) -> None:
    state.sink_blocks.clear()
    state.recent_blocks.clear()
    state.mid_blocks.clear()
    state.next_block_id = 0
    state.total_committed_blocks = 0
    state.batch_size = None
    _invalidate_mid_selection_cache(state)


def commit_fullres_block(
    state: PackLayerCacheState,
    k: torch.Tensor,
    v: torch.Tensor,
    start_frame: int,
    compressor: PackBlockCompressor,
    precomputed_mid: CompressedKVBlock | None = None,
    source_latent: torch.Tensor | None = None,
) -> None:
    """Commit a generated block into sink/recent/mid history."""

    _validate_kv_tensor_pair(k, v)
    _maybe_set_batch_size(state, k.shape[0])
    block = _build_fullres_block(
        state=state,
        k=k,
        v=v,
        start_frame=start_frame,
    )
    if precomputed_mid is not None:
        precomputed_mid = _normalize_precomputed_mid_block(block, precomputed_mid)
        block = FullResKVBlock(k=block.k, v=block.v, meta=block.meta, precomputed_mid=precomputed_mid)
    _maybe_upsert_source_block(state, block.meta, source_latent)

    if len(state.sink_blocks) < state.cfg.sink_blocks:
        state.sink_blocks.append(block)
    else:
        state.recent_blocks.append(block)
        while len(state.recent_blocks) > state.cfg.recent_blocks:
            oldest_recent = state.recent_blocks.popleft()
            if oldest_recent.precomputed_mid is not None:
                compressed = oldest_recent.precomputed_mid
            else:
                if state.cfg.compress_mode == "hr_spatial":
                    raise RuntimeError(
                        "PackForcing hr_spatial compression expected a precomputed mid block, "
                        "but the recent block did not carry one."
                    )
                compressed = compressor(oldest_recent)
            _validate_compressed_block(oldest_recent, compressed)
            state.mid_blocks.append(compressed)
            _evict_mid_blocks_if_needed(state)

    state.total_committed_blocks += 1
    _invalidate_mid_selection_cache(state)


def get_mid_candidate_blocks(state: PackLayerCacheState) -> list[CompressedKVBlock]:
    return list(state.mid_blocks)


def select_mid_blocks(
    state: PackLayerCacheState,
    block_scores: torch.Tensor,
) -> torch.Tensor:
    """Select top-k compressed mid blocks for each batch item."""

    mid_blocks = list(state.mid_blocks)
    num_mid_blocks = len(mid_blocks)
    batch_size = state.batch_size or block_scores.shape[0]

    if num_mid_blocks == 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=block_scores.device)

    if block_scores.ndim != 2:
        raise ValueError(f"block_scores must have shape [B, M], got {tuple(block_scores.shape)}.")
    if block_scores.shape[0] != batch_size:
        raise ValueError(f"Expected block_scores batch size {batch_size}, got {block_scores.shape[0]}.")
    if block_scores.shape[1] != num_mid_blocks:
        raise ValueError(f"Expected {num_mid_blocks} mid scores, got {block_scores.shape[1]}.")

    topk = min(state.cfg.mid_select_topk_blocks, num_mid_blocks)
    if topk == 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=block_scores.device)

    _, selected = torch.topk(block_scores, k=topk, dim=1, largest=True, sorted=False)
    starts = torch.tensor(
        [block.meta.start_frame for block in mid_blocks],
        dtype=torch.long,
        device=block_scores.device,
    )
    selected_starts = starts[selected]
    order = torch.argsort(selected_starts, dim=1)
    return torch.gather(selected, dim=1, index=order)


def maybe_reuse_or_select_mid_blocks(
    state: PackLayerCacheState,
    block_scores: torch.Tensor | None,
    current_start_frame: int,
) -> torch.Tensor:
    """Reuse the current block's mid selection when configured to do so."""

    if state.cfg.reuse_mid_selection_within_block:
        if (
            state.cached_selected_mid_indices is not None
            and state.cached_selection_for_start_frame == current_start_frame
        ):
            return state.cached_selected_mid_indices

    if block_scores is None:
        raise ValueError("block_scores must be provided when no cached mid selection is available.")

    selected = select_mid_blocks(state, block_scores)
    if state.cfg.reuse_mid_selection_within_block:
        state.cached_selected_mid_indices = selected
        state.cached_selection_for_start_frame = current_start_frame
    return selected


def build_attention_view(
    state: PackLayerCacheState,
    anchor_end_frame: int,
    selected_mid_indices: torch.Tensor | None = None,
) -> PackAttentionView:
    """Build the packed history view used by one attention call."""

    if state.batch_size is None:
        raise ValueError("Cannot build an attention view from an uninitialized cache.")

    batch_size = state.batch_size
    reference = _find_reference_tensor(state)
    sink_k, sink_v = _cat_blocks([block.k for block in state.sink_blocks], [block.v for block in state.sink_blocks], reference)
    recent_k, recent_v = _cat_blocks(
        [block.k for block in state.recent_blocks],
        [block.v for block in state.recent_blocks],
        reference,
    )

    mid_blocks = list(state.mid_blocks)
    if not mid_blocks:
        mid_k = _empty_history_like(reference)
        mid_v = _empty_history_like(reference)
        ordered_block_meta_per_batch = [
            [block.meta for block in state.sink_blocks] + [block.meta for block in state.recent_blocks]
            for _ in range(batch_size)
        ]
    else:
        if selected_mid_indices is None:
            selected_mid_indices = _default_mid_selection(state, batch_size, reference.device)
        mid_k, mid_v, selected_mid_meta = _gather_selected_mid_blocks(mid_blocks, selected_mid_indices)
        ordered_block_meta_per_batch = []
        sink_meta = [block.meta for block in state.sink_blocks]
        recent_meta = [block.meta for block in state.recent_blocks]
        for batch_idx in range(batch_size):
            ordered_block_meta_per_batch.append(
                sink_meta + selected_mid_meta[batch_idx] + recent_meta
            )

    history_k = torch.cat([sink_k, mid_k, recent_k], dim=1)
    history_v = torch.cat([sink_v, mid_v, recent_v], dim=1)
    packed_start_frames_per_batch = [
        _build_packed_start_frames(meta_list, anchor_end_frame)
        for meta_list in ordered_block_meta_per_batch
    ]
    expected_query_start_frames = [
        _compute_expected_query_start_frame(anchor_end_frame)
        for meta_list in ordered_block_meta_per_batch
    ]

    return PackAttentionView(
        history_k=history_k,
        history_v=history_v,
        sink_token_count=sink_k.shape[1],
        mid_token_count=mid_k.shape[1],
        recent_token_count=recent_k.shape[1],
        ordered_block_meta_per_batch=ordered_block_meta_per_batch,
        packed_start_frames_per_batch=packed_start_frames_per_batch,
        expected_query_start_frames=expected_query_start_frames,
    )


def get_pack_cache_stats(state: PackLayerCacheState) -> dict:
    sink_tokens = sum(block.meta.num_tokens for block in state.sink_blocks)
    mid_tokens = sum(block.meta.num_tokens for block in state.mid_blocks)
    recent_tokens = sum(block.meta.num_tokens for block in state.recent_blocks)
    return {
        "num_sink_blocks": len(state.sink_blocks),
        "num_mid_blocks": len(state.mid_blocks),
        "num_recent_blocks": len(state.recent_blocks),
        "sink_latents": sum(block.meta.num_frames for block in state.sink_blocks),
        "mid_latents": sum(block.meta.num_frames for block in state.mid_blocks),
        "recent_latents": sum(block.meta.num_frames for block in state.recent_blocks),
        "sink_tokens": sink_tokens,
        "mid_tokens": mid_tokens,
        "recent_tokens": recent_tokens,
        "mid_bank_capacity_blocks": state.cfg.mid_bank_capacity_blocks,
        "mid_select_topk_blocks": state.cfg.mid_select_topk_blocks,
        "total_committed_blocks": state.total_committed_blocks,
    }


def _adaptive_avg_pool_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    batch_size, num_tokens, num_heads, head_dim = x.shape
    if target_tokens == num_tokens:
        return x.clone()

    pooled = x.permute(0, 2, 3, 1).contiguous().reshape(
        batch_size * num_heads * head_dim,
        1,
        num_tokens,
    )
    pooled = F.adaptive_avg_pool1d(pooled, target_tokens)
    pooled = pooled.reshape(batch_size, num_heads, head_dim, target_tokens).permute(0, 3, 1, 2)
    return pooled.contiguous()


def _build_fullres_block(
    state: PackLayerCacheState,
    k: torch.Tensor,
    v: torch.Tensor,
    start_frame: int,
) -> FullResKVBlock:
    cfg = state.cfg
    num_tokens = k.shape[1]
    if num_tokens % cfg.frame_seq_len != 0:
        raise ValueError(
            f"Expected token count divisible by frame_seq_len={cfg.frame_seq_len}, got {num_tokens}."
        )
    num_frames = num_tokens // cfg.frame_seq_len
    meta = BlockMeta(
        block_id=state.next_block_id,
        start_frame=start_frame,
        num_frames=num_frames,
        packed_frame_span=num_frames,
        tokens_per_frame=cfg.frame_seq_len,
        num_tokens=num_tokens,
        is_compressed=False,
        source_block_id=None,
    )
    state.next_block_id += 1
    return FullResKVBlock(k=k, v=v, meta=meta)


def _normalize_precomputed_mid_block(
    source_block: FullResKVBlock,
    precomputed_mid: CompressedKVBlock,
) -> CompressedKVBlock:
    _validate_kv_tensor_pair(precomputed_mid.k, precomputed_mid.v)
    if precomputed_mid.k.shape[0] != source_block.k.shape[0]:
        raise ValueError("Precomputed mid block batch size must match its source full-res block.")
    if precomputed_mid.meta.num_frames != source_block.meta.num_frames:
        raise ValueError("Precomputed mid block must preserve source num_frames.")
    if precomputed_mid.meta.packed_frame_span != source_block.meta.packed_frame_span:
        raise ValueError("Precomputed mid block must preserve source packed_frame_span.")
    if precomputed_mid.k.shape[1] % source_block.meta.num_frames != 0:
        raise ValueError(
            "Precomputed mid block token count must be divisible by source num_frames."
        )

    normalized_meta = BlockMeta(
        block_id=source_block.meta.block_id,
        start_frame=source_block.meta.start_frame,
        num_frames=source_block.meta.num_frames,
        packed_frame_span=source_block.meta.packed_frame_span,
        tokens_per_frame=precomputed_mid.k.shape[1] // source_block.meta.num_frames,
        num_tokens=precomputed_mid.k.shape[1],
        is_compressed=True,
        source_block_id=source_block.meta.block_id,
    )
    return CompressedKVBlock(
        k=precomputed_mid.k,
        v=precomputed_mid.v,
        meta=normalized_meta,
    )


def _maybe_upsert_source_block(
    state: PackLayerCacheState,
    block_meta: BlockMeta,
    source_latent: torch.Tensor | None,
) -> None:
    del state, block_meta, source_latent


def _evict_mid_blocks_if_needed(state: PackLayerCacheState) -> None:
    if state.cfg.evict_mode != "fifo":
        raise ValueError(f"Unsupported evict_mode={state.cfg.evict_mode}.")
    while len(state.mid_blocks) > state.cfg.mid_bank_capacity_blocks:
        state.mid_blocks.popleft()


def _default_mid_selection(
    state: PackLayerCacheState,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    num_mid_blocks = len(state.mid_blocks)
    topk = min(state.cfg.mid_select_topk_blocks, num_mid_blocks)
    if topk == 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    indices = torch.arange(topk, device=device, dtype=torch.long)
    return indices.unsqueeze(0).expand(batch_size, -1).clone()


def _gather_selected_mid_blocks(
    mid_blocks: list[CompressedKVBlock],
    selected_mid_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[list[BlockMeta]]]:
    if selected_mid_indices.ndim != 2:
        raise ValueError(
            f"selected_mid_indices must have shape [B, K], got {tuple(selected_mid_indices.shape)}."
        )

    batch_size = selected_mid_indices.shape[0]
    token_counts = {block.k.shape[1] for block in mid_blocks}
    if len(token_counts) != 1:
        raise ValueError("All compressed mid blocks must share the same token count in the current skeleton.")

    stacked_k = torch.stack([block.k for block in mid_blocks], dim=1)  # [B, M, Nc, H, D]
    stacked_v = torch.stack([block.v for block in mid_blocks], dim=1)
    stacked_batch_size, num_mid_blocks, num_tokens, num_heads, head_dim = stacked_k.shape
    if stacked_batch_size != batch_size:
        raise ValueError(
            f"selected_mid_indices batch size {batch_size} does not match cached batch size {stacked_batch_size}."
        )
    if selected_mid_indices.numel() == 0:
        empty = stacked_k.new_zeros((batch_size, 0, num_heads, head_dim))
        return empty, empty, [[] for _ in range(batch_size)]

    if torch.any(selected_mid_indices < 0) or torch.any(selected_mid_indices >= num_mid_blocks):
        raise IndexError("selected_mid_indices contains out-of-range mid block indices.")

    selected_meta: list[list[BlockMeta]] = []
    batch_indices = torch.arange(batch_size, device=selected_mid_indices.device).unsqueeze(1)
    gathered_k = stacked_k[batch_indices, selected_mid_indices]  # [B, K, Nc, H, D]
    gathered_v = stacked_v[batch_indices, selected_mid_indices]

    for batch_idx in range(batch_size):
        selected_meta.append([mid_blocks[i].meta for i in selected_mid_indices[batch_idx].tolist()])

    gathered_k_tensor = gathered_k.reshape(batch_size, -1, num_heads, head_dim)
    gathered_v_tensor = gathered_v.reshape(batch_size, -1, num_heads, head_dim)
    return gathered_k_tensor, gathered_v_tensor, selected_meta


def _build_packed_start_frames(
    meta_list: list[BlockMeta],
    anchor_end_frame: int,
) -> list[int]:
    packed_positions = [0] * len(meta_list)
    cursor = int(anchor_end_frame)

    for i in range(len(meta_list) - 1, -1, -1):
        meta = meta_list[i]
        cursor -= meta.packed_frame_span
        packed_positions[i] = cursor
    return packed_positions


def _compute_expected_query_start_frame(anchor_end_frame: int) -> int:
    return int(anchor_end_frame)


def _cat_blocks(
    k_blocks: list[torch.Tensor],
    v_blocks: list[torch.Tensor],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(k_blocks) != len(v_blocks):
        raise ValueError("k_blocks and v_blocks must have the same length.")
    if not k_blocks:
        empty = _empty_history_like(reference)
        return empty, empty
    return torch.cat(k_blocks, dim=1), torch.cat(v_blocks, dim=1)


def _empty_history_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros((reference.shape[0], 0, reference.shape[2], reference.shape[3]))


def _find_reference_tensor(state: PackLayerCacheState) -> torch.Tensor:
    for block in state.sink_blocks:
        return block.k
    for block in state.recent_blocks:
        return block.k
    for block in state.mid_blocks:
        return block.k
    raise ValueError("Pack cache has no tensors yet.")


def _validate_kv_tensor_pair(k: torch.Tensor, v: torch.Tensor) -> None:
    if k.shape != v.shape:
        raise ValueError(f"k and v must share the same shape, got {tuple(k.shape)} vs {tuple(v.shape)}.")
    if k.ndim != 4:
        raise ValueError(f"Expected kv tensors shaped [B, N, H, D], got {tuple(k.shape)}.")


def _validate_compressed_block(
    source_block: FullResKVBlock,
    compressed_block: CompressedKVBlock,
) -> None:
    _validate_kv_tensor_pair(compressed_block.k, compressed_block.v)
    if compressed_block.meta.start_frame != source_block.meta.start_frame:
        raise ValueError("Compressed block must preserve source start_frame.")
    if compressed_block.meta.num_frames != source_block.meta.num_frames:
        raise ValueError("Compressed block must preserve source num_frames.")
    if compressed_block.meta.packed_frame_span != source_block.meta.packed_frame_span:
        raise ValueError("Compressed block must preserve source packed_frame_span in the current skeleton.")
    if not compressed_block.meta.is_compressed:
        raise ValueError("Compressed block meta must set is_compressed=True.")


def _maybe_set_batch_size(state: PackLayerCacheState, batch_size: int) -> None:
    if state.batch_size is None:
        state.batch_size = batch_size
    elif state.batch_size != batch_size:
        raise ValueError(f"Pack cache batch size mismatch: expected {state.batch_size}, got {batch_size}.")


def _invalidate_mid_selection_cache(state: PackLayerCacheState) -> None:
    state.cached_selected_mid_indices = None
    state.cached_selection_for_start_frame = None
