from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from wan.modules.attention import attention
from wan.modules.causal_model import (
    CausalWanAttentionBlock,
    CausalWanModel,
    CausalWanSelfAttention,
    causal_rope_apply,
)
from wan.modules.model import WAN_CROSSATTENTION_CLASSES, WanLayerNorm
from wan.modules.pack_cache import (
    BlockMeta,
    CompressedKVBlock,
    FullResKVBlock,
    IdentityCompressor,
    PackBlockCompressor,
    PackCacheConfig,
    PackLayerCacheState,
    TokenAverageCompressor,
    build_attention_view,
    commit_fullres_block,
    get_mid_candidate_blocks,
    init_empty_pack_layer_cache,
    maybe_reuse_or_select_mid_blocks,
)


@dataclass
class PackConfig:
    """PackForcing-specific cache configuration used by the model layer."""

    enable: bool = False
    sink_blocks: int = 2
    recent_blocks: int = 1
    mid_bank_capacity_blocks: int = 3
    mid_select_topk_blocks: int = 2
    mid_selection_mode: str = "recency"
    compress_mode: str = "token_avg_pool"
    compressed_tokens_per_block: int = 1560
    evict_mode: str = "fifo"
    reuse_mid_selection_within_block: bool = True
    enable_rope_adjustment: bool = False

    def __post_init__(self) -> None:
        valid_mid_selection_modes = {"recency", "query_score"}
        if self.mid_selection_mode not in valid_mid_selection_modes:
            raise ValueError(
                f"Unsupported mid_selection_mode={self.mid_selection_mode}. "
                f"Expected one of {sorted(valid_mid_selection_modes)}."
            )

    def build_cache_config(self, frame_seq_len: int, num_frame_per_block: int) -> PackCacheConfig:
        return PackCacheConfig(
            frame_seq_len=frame_seq_len,
            num_frame_per_block=num_frame_per_block,
            sink_blocks=self.sink_blocks,
            recent_blocks=self.recent_blocks,
            mid_bank_capacity_blocks=self.mid_bank_capacity_blocks,
            mid_select_topk_blocks=self.mid_select_topk_blocks,
            compress_mode=self.compress_mode,
            evict_mode=self.evict_mode,
            reuse_mid_selection_within_block=self.reuse_mid_selection_within_block,
            enable_rope_adjustment=self.enable_rope_adjustment,
        )


def init_pack_kv_cache(
    num_layers: int,
    pack_cfg: PackCacheConfig,
) -> list[PackLayerCacheState]:
    return [init_empty_pack_layer_cache(pack_cfg) for _ in range(num_layers)]


class CausalWanSelfAttentionPackForcing(CausalWanSelfAttention):
    """Causal self-attention with a block-level PackForcing history backend."""

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        pack_cfg: PackConfig | None = None,
    ):
        super().__init__(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.pack_cfg = pack_cfg or PackConfig(enable=False)
        self._compressor: PackBlockCompressor | None = None

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        if kv_cache is None or not self.pack_cfg.enable:
            return super().forward(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=cache_start,
            )

        if not isinstance(kv_cache, PackLayerCacheState):
            raise TypeError(
                "PackForcing attention expects each kv_cache item to be a PackLayerCacheState."
            )

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        frame_seqlen = int(torch.prod(grid_sizes[0][1:]).item())
        current_start_frame = current_start // frame_seqlen
        current_num_frames = s // frame_seqlen
        absolute_roped_query = causal_rope_apply(
            q,
            grid_sizes,
            freqs,
            start_frame=current_start_frame,
        ).type_as(v)
        absolute_roped_key = causal_rope_apply(
            k,
            grid_sizes,
            freqs,
            start_frame=current_start_frame,
        ).type_as(v)

        tail = _get_tail_fullres_block(kv_cache)
        has_same_block_cached = tail is not None and tail.meta.start_frame == current_start_frame
        has_any_history = bool(kv_cache.sink_blocks or kv_cache.recent_blocks or kv_cache.mid_blocks)

        if not has_any_history:
            x = attention(absolute_roped_query, absolute_roped_key, v)
            compressor = self._get_or_create_compressor()
            _upsert_pack_cache_block(
                kv_cache,
                absolute_roped_key,
                v,
                start_frame=current_start_frame,
                compressor=compressor,
            )
            x = x.flatten(2)
            x = self.o(x)
            return x

        if has_same_block_cached:
            compressor = self._get_or_create_compressor()
            _upsert_pack_cache_block(
                kv_cache,
                absolute_roped_key,
                v,
                start_frame=current_start_frame,
                compressor=compressor,
            )

        selected_mid_indices = None
        mid_blocks = get_mid_candidate_blocks(kv_cache)
        if mid_blocks:
            block_scores = _compute_mid_block_scores(
                absolute_roped_query,
                mid_blocks,
                mode=self.pack_cfg.mid_selection_mode,
            )
            selected_mid_indices = maybe_reuse_or_select_mid_blocks(
                kv_cache,
                block_scores=block_scores,
                current_start_frame=current_start_frame,
            )

        anchor_end_frame = current_start_frame if not has_same_block_cached else current_start_frame + current_num_frames
        history_view = build_attention_view(
            kv_cache,
            anchor_end_frame=anchor_end_frame,
            selected_mid_indices=selected_mid_indices,
        )
        if kv_cache.cfg.enable_rope_adjustment:
            history_key = _apply_packed_rope_to_history_k(
                history_view.history_k,
                history_view.ordered_block_meta_per_batch,
                history_view.packed_start_frames_per_batch,
                freqs,
            )
        else:
            history_key = history_view.history_k

        attn_query = absolute_roped_query
        current_attn_key = absolute_roped_key

        if has_same_block_cached:
            all_key = history_key
            all_val = history_view.history_v
        else:
            all_key = torch.cat([history_key, current_attn_key], dim=1)
            all_val = torch.cat([history_view.history_v, v], dim=1)
        x = attention(attn_query, all_key, all_val)

        if not has_same_block_cached:
            compressor = self._get_or_create_compressor()
            _upsert_pack_cache_block(
                kv_cache,
                absolute_roped_key,
                v,
                start_frame=current_start_frame,
                compressor=compressor,
            )

        x = x.flatten(2)
        x = self.o(x)
        return x

    def _get_or_create_compressor(self) -> PackBlockCompressor:
        if self._compressor is not None:
            return self._compressor

        if self.pack_cfg.compress_mode == "identity":
            self._compressor = IdentityCompressor()
        elif self.pack_cfg.compress_mode == "token_avg_pool":
            self._compressor = TokenAverageCompressor(
                target_tokens_per_block=self.pack_cfg.compressed_tokens_per_block
            )
        else:
            raise ValueError(f"Unsupported pack compress_mode={self.pack_cfg.compress_mode}.")
        return self._compressor


class CausalWanAttentionBlockPackForcing(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        pack_cfg: PackConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttentionPackForcing(
            dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            eps,
            pack_cfg=pack_cfg,
        )
        self.norm3 = WanLayerNorm(
            dim,
            eps,
            elementwise_affine=True,
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
    ):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_lens,
            crossattn_cache=crossattn_cache,
        )
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


class CausalWanModelPackForcing(CausalWanModel):
    ignore_for_config = CausalWanModel.ignore_for_config
    _no_split_modules = ['CausalWanAttentionBlockPackForcing']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        pack_enable: bool = False,
        pack_sink_blocks: int = 2,
        pack_recent_blocks: int = 1,
        pack_mid_bank_capacity_blocks: int = 3,
        pack_mid_select_topk_blocks: int = 2,
        pack_mid_selection_mode: str = "recency",
        pack_compress_mode: str = "token_avg_pool",
        pack_compressed_tokens_per_block: int = 1560,
        pack_evict_mode: str = "fifo",
        pack_reuse_mid_selection_within_block: bool = True,
        pack_enable_rope_adjustment: bool = False,
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.pack_cfg = PackConfig(
            enable=pack_enable,
            sink_blocks=pack_sink_blocks,
            recent_blocks=pack_recent_blocks,
            mid_bank_capacity_blocks=pack_mid_bank_capacity_blocks,
            mid_select_topk_blocks=pack_mid_select_topk_blocks,
            mid_selection_mode=pack_mid_selection_mode,
            compress_mode=pack_compress_mode,
            compressed_tokens_per_block=pack_compressed_tokens_per_block,
            evict_mode=pack_evict_mode,
            reuse_mid_selection_within_block=pack_reuse_mid_selection_within_block,
            enable_rope_adjustment=pack_enable_rope_adjustment,
        )

        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlockPackForcing(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                local_attn_size,
                sink_size,
                qk_norm,
                cross_attn_norm,
                eps,
                pack_cfg=self.pack_cfg,
            )
            for _ in range(num_layers)
        ])
        self.init_weights()

    def make_pack_kv_cache(self) -> list[PackLayerCacheState]:
        pack_cache_cfg = self.pack_cfg.build_cache_config(
            frame_seq_len=1560,
            num_frame_per_block=self.num_frame_per_block,
        )
        return init_pack_kv_cache(self.num_layers, pack_cache_cfg)


def _compute_mid_block_scores(
    roped_query: torch.Tensor,
    mid_blocks: list[CompressedKVBlock],
    mode: str,
) -> torch.Tensor:
    """Compute debugging-friendly block scores for mid selection."""

    if mode == "recency":
        del roped_query
        batch_size = mid_blocks[0].k.shape[0]
        device = mid_blocks[0].k.device
        num_mid_blocks = len(mid_blocks)
        return torch.arange(
            num_mid_blocks,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(batch_size, -1)

    if mode == "query_score":
        query_summary = roped_query.mean(dim=1)  # [B, H, D]
        stacked_mid_summary = torch.stack(
            [block.k.mean(dim=1) for block in mid_blocks],
            dim=1,
        )  # [B, M, H, D]
        return (query_summary.unsqueeze(1) * stacked_mid_summary).sum(dim=-1).mean(dim=-1)

    raise ValueError(f"Unsupported mid selection mode: {mode}.")


def _upsert_pack_cache_block(
    state: PackLayerCacheState,
    roped_key: torch.Tensor,
    value: torch.Tensor,
    start_frame: int,
    compressor: PackBlockCompressor,
) -> None:
    """Update the current block in-place across denoising steps, or append a new block."""

    tail = _get_tail_fullres_block(state)
    if tail is not None and tail.meta.start_frame == start_frame:
        updated = FullResKVBlock(k=roped_key, v=value, meta=tail.meta)
        if state.recent_blocks:
            state.recent_blocks[-1] = updated
        else:
            state.sink_blocks[-1] = updated
        state.cached_selected_mid_indices = None
        state.cached_selection_for_start_frame = None
        return

    commit_fullres_block(
        state,
        roped_key,
        value,
        start_frame=start_frame,
        compressor=compressor,
    )


def _get_tail_fullres_block(state: PackLayerCacheState) -> FullResKVBlock | None:
    if state.recent_blocks:
        return state.recent_blocks[-1]
    if state.sink_blocks:
        return state.sink_blocks[-1]
    return None


def _apply_packed_rope_to_history_k(
    history_k: torch.Tensor,
    ordered_block_meta_per_batch: list[list[BlockMeta]],
    packed_start_frames_per_batch: list[list[int]],
    freqs: torch.Tensor,
) -> torch.Tensor:
    adjusted_history_k = history_k.clone()

    for batch_idx, (meta_list, packed_starts) in enumerate(
        zip(ordered_block_meta_per_batch, packed_start_frames_per_batch)
    ):
        token_cursor = 0
        for meta, packed_start_frame in zip(meta_list, packed_starts):
            block_token_count = meta.num_tokens
            next_cursor = token_cursor + block_token_count
            adjusted_history_k[batch_idx, token_cursor:next_cursor] = _rebase_block_temporal_rope(
                adjusted_history_k[batch_idx, token_cursor:next_cursor],
                meta,
                packed_start_frame,
                freqs,
            )
            token_cursor = next_cursor

        if token_cursor != history_k.shape[1]:
            raise ValueError(
                "Packed history metadata does not match concatenated history tensor length."
            )

    return adjusted_history_k


def _rebase_block_temporal_rope(
    block_k: torch.Tensor,
    meta: BlockMeta,
    packed_start_frame: int,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Shift one cached block along the temporal RoPE axis by a uniform phase delta.

    This implementation assumes the cached keys were already RoPE'd in absolute
    time before any token-linear block compression (for example identity or
    average pooling along the token axis). Under that assumption, moving the
    whole block by delta_t frames is equivalent to multiplying every token by
    the same temporal complex phase factor.
    """

    if meta.start_frame == packed_start_frame:
        return block_k
    if block_k.shape[0] != meta.num_tokens:
        raise ValueError(
            f"Block tensor length {block_k.shape[0]} does not match meta.num_tokens={meta.num_tokens}."
        )

    num_heads = block_k.shape[1]
    head_dim = block_k.shape[2]
    if head_dim % 2 != 0:
        raise ValueError(f"Expected even head_dim for RoPE, got {head_dim}.")

    num_frames = meta.packed_frame_span
    time_complex_dim = (head_dim // 2) - 2 * ((head_dim // 2) // 3)
    if time_complex_dim <= 0:
        return block_k

    freqs_t = freqs[:, :time_complex_dim]
    delta_t = int(packed_start_frame) - meta.start_frame
    delta_index = abs(delta_t)
    if delta_index >= freqs_t.shape[0]:
        raise ValueError(
            "RoPE frequency table is too short for PackForcing phase shift: "
            f"delta_t={delta_t}, max_delta={freqs_t.shape[0] - 1}."
        )

    phase_shift = freqs_t[delta_index]
    if delta_t < 0:
        phase_shift = phase_shift.conj()

    block_complex = torch.view_as_complex(
        block_k.to(torch.float64).reshape(meta.num_tokens, num_heads, -1, 2)
    )
    rotated_time = block_complex[..., :time_complex_dim] * phase_shift
    block_complex = torch.cat([rotated_time, block_complex[..., time_complex_dim:]], dim=-1)
    return torch.view_as_real(block_complex).flatten(2).type_as(block_k)
