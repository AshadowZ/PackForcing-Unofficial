from __future__ import annotations

from collections import deque
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
from wan.modules.model import WAN_CROSSATTENTION_CLASSES, WanLayerNorm, sinusoidal_embedding_1d
from wan.modules.pack_compressor import PackHRSpatialCompressor
from wan.modules.pack_cache import (
    BlockMeta,
    CompressedKVBlock,
    FullResKVBlock,
    IdentityCompressor,
    PackBlockCompressor,
    PackCacheConfig,
    PackLayerCacheState,
    PackSourceCacheState,
    SourceLatentBlock,
    TokenAverageCompressor,
    build_attention_view,
    commit_fullres_block,
    get_mid_candidate_blocks,
    init_empty_pack_layer_cache,
    maybe_reuse_or_select_mid_blocks,
    reset_pack_layer_cache,
    select_mid_blocks,
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
    compressor_detach_inputs: bool = True
    evict_mode: str = "fifo"
    reuse_mid_selection_within_block: bool = True
    enable_rope_adjustment: bool = False

    def __post_init__(self) -> None:
        valid_mid_selection_modes = {"recency", "query_score"}
        valid_compress_modes = {"identity", "token_avg_pool", "hr_spatial"}
        if self.mid_selection_mode not in valid_mid_selection_modes:
            raise ValueError(
                f"Unsupported mid_selection_mode={self.mid_selection_mode}. "
                f"Expected one of {sorted(valid_mid_selection_modes)}."
            )
        if self.compress_mode not in valid_compress_modes:
            raise ValueError(
                f"Unsupported compress_mode={self.compress_mode}. "
                f"Expected one of {sorted(valid_compress_modes)}."
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
        self._pack_history_latent_compressor = None

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
        pack_compressed_hidden=None,
        pack_compressed_grid_sizes=None,
        pack_history_mid_latents=None,
        pack_history_mid_hidden=None,
        pack_history_mid_grid_sizes=None,
        pack_source_latent=None,
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
        allow_cache_mutation = not torch.is_grad_enabled()
        history_state = kv_cache
        effective_has_same_block_cached = has_same_block_cached
        if has_same_block_cached and not allow_cache_mutation:
            history_state = _history_state_without_current_block(kv_cache, current_start_frame)
            effective_has_same_block_cached = False
        effective_has_any_history = bool(
            history_state.sink_blocks or history_state.recent_blocks or history_state.mid_blocks
        )

        if not effective_has_any_history:
            x = attention(absolute_roped_query, absolute_roped_key, v)
            if allow_cache_mutation:
                compressor = self._get_or_create_compressor()
                precomputed_mid = self._build_precomputed_mid_block(
                    compressed_hidden=pack_compressed_hidden,
                    compressed_grid_sizes=pack_compressed_grid_sizes,
                    freqs=freqs,
                    current_start_frame=current_start_frame,
                    current_num_frames=current_num_frames,
                )
                _upsert_pack_cache_block(
                    kv_cache,
                    absolute_roped_key,
                    v,
                    start_frame=current_start_frame,
                    compressor=compressor,
                    precomputed_mid=precomputed_mid,
                    source_latent=pack_source_latent,
                )
            x = x.flatten(2)
            x = self.o(x)
            return x

        if has_same_block_cached and allow_cache_mutation:
            compressor = self._get_or_create_compressor()
            precomputed_mid = self._build_precomputed_mid_block(
                compressed_hidden=pack_compressed_hidden,
                compressed_grid_sizes=pack_compressed_grid_sizes,
                freqs=freqs,
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
            )
            _upsert_pack_cache_block(
                kv_cache,
                absolute_roped_key,
                v,
                start_frame=current_start_frame,
                compressor=compressor,
                precomputed_mid=precomputed_mid,
                source_latent=pack_source_latent,
            )

        selected_mid_indices = None
        mid_blocks = get_mid_candidate_blocks(kv_cache)
        if mid_blocks:
            block_scores = _compute_mid_block_scores(
                absolute_roped_query,
                mid_blocks,
                mode=self.pack_cfg.mid_selection_mode,
            )
            if allow_cache_mutation:
                selected_mid_indices = maybe_reuse_or_select_mid_blocks(
                    kv_cache,
                    block_scores=block_scores,
                    current_start_frame=current_start_frame,
                )
            else:
                selected_mid_indices = select_mid_blocks(history_state, block_scores)

        anchor_end_frame = (
            current_start_frame
            if not effective_has_same_block_cached
            else current_start_frame + current_num_frames
        )
        history_view_state = history_state
        if (
            not allow_cache_mutation
            and self.pack_cfg.compress_mode == "hr_spatial"
            and pack_history_mid_hidden is not None
        ):
            trainable_mid_blocks = self._build_history_mid_blocks_from_hidden(
                history_state=history_state,
                compressed_hidden=pack_history_mid_hidden,
                compressed_grid_sizes=pack_history_mid_grid_sizes,
                freqs=freqs,
            )
            history_view_state = _state_with_mid_blocks(history_state, trainable_mid_blocks)
        history_view = build_attention_view(
            history_view_state,
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

        if effective_has_same_block_cached:
            all_key = history_key
            all_val = history_view.history_v
        else:
            all_key = torch.cat([history_key, current_attn_key], dim=1)
            all_val = torch.cat([history_view.history_v, v], dim=1)
        x = attention(attn_query, all_key, all_val)

        if not effective_has_same_block_cached and allow_cache_mutation:
            compressor = self._get_or_create_compressor()
            precomputed_mid = self._build_precomputed_mid_block(
                compressed_hidden=pack_compressed_hidden,
                compressed_grid_sizes=pack_compressed_grid_sizes,
                freqs=freqs,
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
            )
            _upsert_pack_cache_block(
                kv_cache,
                absolute_roped_key,
                v,
                start_frame=current_start_frame,
                compressor=compressor,
                precomputed_mid=precomputed_mid,
                source_latent=pack_source_latent,
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
        elif self.pack_cfg.compress_mode == "hr_spatial":
            # hr_spatial mid blocks are precomputed from shared latent-space hidden tokens.
            self._compressor = IdentityCompressor()
        else:
            raise ValueError(f"Unsupported pack compress_mode={self.pack_cfg.compress_mode}.")
        return self._compressor

    def _build_precomputed_mid_block(
        self,
        compressed_hidden: torch.Tensor | None,
        compressed_grid_sizes: torch.Tensor | None,
        freqs: torch.Tensor,
        current_start_frame: int,
        current_num_frames: int,
    ) -> CompressedKVBlock | None:
        if self.pack_cfg.compress_mode != "hr_spatial":
            return None
        if compressed_hidden is None or compressed_grid_sizes is None:
            raise RuntimeError(
                "PackForcing hr_spatial compression expected transient compressed hidden tokens."
            )

        batch_size, token_count, _ = compressed_hidden.shape
        if compressed_grid_sizes.ndim != 2 or compressed_grid_sizes.shape[1] != 3:
            raise ValueError(
                "compressed_grid_sizes must have shape [B, 3] for hr_spatial compression."
            )
        if compressed_grid_sizes.shape[0] != batch_size:
            raise ValueError("compressed_grid_sizes batch size must match compressed_hidden.")
        if torch.any(compressed_grid_sizes[:, 0] != current_num_frames):
            raise ValueError(
                "hr_spatial compressor must preserve the temporal axis of the current block."
            )
        if token_count % current_num_frames != 0:
            raise ValueError(
                "hr_spatial compressed token count must be divisible by current_num_frames."
            )

        compressed_key = self.norm_k(self.k(compressed_hidden)).view(
            batch_size, token_count, self.num_heads, self.head_dim
        )
        compressed_value = self.v(compressed_hidden).view(
            batch_size, token_count, self.num_heads, self.head_dim
        )
        compressed_key = causal_rope_apply(
            compressed_key,
            compressed_grid_sizes,
            freqs,
            start_frame=current_start_frame,
        ).type_as(compressed_value)

        meta = BlockMeta(
            block_id=-1,
            start_frame=current_start_frame,
            num_frames=current_num_frames,
            packed_frame_span=current_num_frames,
            tokens_per_frame=token_count // current_num_frames,
            num_tokens=token_count,
            is_compressed=True,
            source_block_id=None,
        )
        return CompressedKVBlock(
            k=compressed_key,
            v=compressed_value,
            meta=meta,
        )

    def _build_history_mid_blocks_from_hidden(
        self,
        history_state: PackLayerCacheState,
        compressed_hidden: torch.Tensor,
        compressed_grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> list[CompressedKVBlock]:
        cached_mid_blocks = get_mid_candidate_blocks(history_state)
        if compressed_hidden.ndim != 4:
            raise ValueError(
                "pack_history_mid_hidden must have shape [M, B, N, D]."
            )
        if compressed_grid_sizes.ndim != 3 or compressed_grid_sizes.shape[-1] != 3:
            raise ValueError(
                "pack_history_mid_grid_sizes must have shape [M, B, 3]."
            )
        if compressed_hidden.shape[0] != len(cached_mid_blocks):
            raise ValueError(
                "pack_history_mid_hidden count must match the number of cached mid blocks."
            )
        if compressed_grid_sizes.shape[:2] != compressed_hidden.shape[:2]:
            raise ValueError(
                "pack_history_mid_grid_sizes must align with pack_history_mid_hidden on [M, B]."
            )

        trainable_mid_blocks: list[CompressedKVBlock] = []
        for block_idx, cached_mid in enumerate(cached_mid_blocks):
            meta = cached_mid.meta
            block_hidden = compressed_hidden[block_idx]
            block_grid_sizes = compressed_grid_sizes[block_idx]
            batch_size, token_count, _ = block_hidden.shape
            if token_count != meta.num_tokens:
                raise ValueError(
                    "Train-time compressed mid token count does not match cached mid metadata."
                )
            block_key = self.norm_k(self.k(block_hidden)).view(
                batch_size, token_count, self.num_heads, self.head_dim
            )
            block_value = self.v(block_hidden).view(
                batch_size, token_count, self.num_heads, self.head_dim
            )
            block_key = causal_rope_apply(
                block_key,
                block_grid_sizes,
                freqs,
                start_frame=meta.start_frame,
            ).type_as(block_value)
            trainable_mid_blocks.append(
                CompressedKVBlock(
                    k=block_key,
                    v=block_value,
                    meta=meta,
                )
            )
        return trainable_mid_blocks


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
        pack_compressed_hidden=None,
        pack_compressed_grid_sizes=None,
        pack_history_mid_latents=None,
        pack_history_mid_hidden=None,
        pack_history_mid_grid_sizes=None,
        pack_source_latent=None,
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
            pack_compressed_hidden,
            pack_compressed_grid_sizes,
            pack_history_mid_latents,
            pack_history_mid_hidden,
            pack_history_mid_grid_sizes,
            pack_source_latent,
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
        pack_compressor_detach_inputs: bool = True,
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
            compressor_detach_inputs=pack_compressor_detach_inputs,
            evict_mode=pack_evict_mode,
            reuse_mid_selection_within_block=pack_reuse_mid_selection_within_block,
            enable_rope_adjustment=pack_enable_rope_adjustment,
        )
        self.pack_compressor = (
            PackHRSpatialCompressor(
                latent_channels=in_dim,
                hidden_dim=dim,
                detach_inputs=self.pack_cfg.compressor_detach_inputs,
            )
            if self.pack_cfg.enable and self.pack_cfg.compress_mode == "hr_spatial"
            else None
        )
        self.pack_layer_cache = self.make_pack_kv_cache()
        self.pack_source_state = PackSourceCacheState()

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

    def get_pack_layer_cache(self) -> list[PackLayerCacheState]:
        if self.pack_layer_cache is None:
            self.pack_layer_cache = self.make_pack_kv_cache()
        return self.pack_layer_cache

    def reset_pack_layer_cache(self) -> None:
        for layer_cache in self.get_pack_layer_cache():
            layer_cache.cfg.num_frame_per_block = self.num_frame_per_block
            reset_pack_layer_cache(layer_cache)

    def reset_pack_source_cache(self) -> None:
        self.pack_source_state.recent_blocks.clear()
        self.pack_source_state.mid_blocks.clear()
        self.pack_source_state.next_block_id = 0
        self.pack_source_state.total_committed_blocks = 0
        self.pack_source_state.batch_size = None
        self.pack_source_state.last_block_start_frame = None

    def _gather_history_mid_latents(
        self,
    ) -> torch.Tensor | None:
        source_state = self.pack_source_state
        if source_state is None or not source_state.mid_blocks:
            return None

        latents = [block.latent for block in source_state.mid_blocks]
        latent_shapes = {tuple(latent.shape) for latent in latents}
        if len(latent_shapes) != 1:
            raise ValueError(
                "PackForcing train-time history mid latents must share the same shape."
            )
        return torch.stack(latents, dim=0)

    def _compress_history_mid_latents(
        self,
        source_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if source_latents is None:
            return None, None
        if self.pack_compressor is None:
            raise RuntimeError("PackForcing hr_spatial training expected pack_compressor.")
        if source_latents.ndim != 6:
            raise ValueError(
                "PackForcing history mid latents must have shape [M, B, C, T, H, W]."
            )

        num_mid_blocks, batch_size, channels, num_frames, height, width = source_latents.shape
        flat_latents = source_latents.reshape(
            num_mid_blocks * batch_size,
            channels,
            num_frames,
            height,
            width,
        )
        compressed_hidden, compressed_grid_sizes = self.pack_compressor.compress_latent(flat_latents)
        compressed_hidden = compressed_hidden.reshape(
            num_mid_blocks,
            batch_size,
            compressed_hidden.shape[1],
            compressed_hidden.shape[2],
        )
        compressed_grid_sizes = compressed_grid_sizes.reshape(num_mid_blocks, batch_size, 3)
        return compressed_hidden, compressed_grid_sizes

    def _commit_source_latent_block(
        self,
        source_latent: torch.Tensor,
        start_frame: int,
    ) -> None:
        state = self.pack_source_state
        if source_latent.ndim != 5:
            raise ValueError(
                "PackForcing source latent blocks must have shape [B, C, T, H, W], "
                f"got {tuple(source_latent.shape)}."
            )
        batch_size, _, num_frames, _, _ = source_latent.shape
        if state.batch_size is None:
            state.batch_size = batch_size
        elif state.batch_size != batch_size:
            raise ValueError(
                f"PackForcing source bank batch size mismatch: expected {state.batch_size}, got {batch_size}."
            )

        if state.last_block_start_frame == start_frame:
            if state.recent_blocks and state.recent_blocks[-1].meta.start_frame == start_frame:
                state.recent_blocks[-1] = SourceLatentBlock(
                    latent=source_latent,
                    meta=state.recent_blocks[-1].meta,
                )
            return

        block_id = state.next_block_id
        meta = BlockMeta(
            block_id=block_id,
            start_frame=start_frame,
            num_frames=num_frames,
            packed_frame_span=num_frames,
            tokens_per_frame=1,
            num_tokens=num_frames,
            is_compressed=False,
            source_block_id=None,
        )
        state.next_block_id += 1
        state.total_committed_blocks += 1
        state.last_block_start_frame = start_frame

        if block_id < self.pack_cfg.sink_blocks:
            return

        state.recent_blocks.append(SourceLatentBlock(latent=source_latent, meta=meta))
        while len(state.recent_blocks) > self.pack_cfg.recent_blocks:
            state.mid_blocks.append(state.recent_blocks.popleft())
            while len(state.mid_blocks) > self.pack_cfg.mid_bank_capacity_blocks:
                state.mid_blocks.popleft()

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
    ):
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None

        kv_cache = self.get_pack_layer_cache()

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        raw_latent_batch = x if isinstance(x, torch.Tensor) else torch.stack(x)
        pack_compressed_hidden = None
        pack_compressed_grid_sizes = None
        pack_history_mid_latents = None
        pack_history_mid_hidden = None
        pack_history_mid_grid_sizes = None
        if (
            kv_cache is not None
            and self.pack_compressor is not None
            and self.pack_cfg.enable
            and self.pack_cfg.compress_mode == "hr_spatial"
        ):
            if torch.is_grad_enabled():
                pack_history_mid_latents = self._gather_history_mid_latents()
                pack_history_mid_hidden, pack_history_mid_grid_sizes = self._compress_history_mid_latents(
                    pack_history_mid_latents
                )
            else:
                pack_compressed_hidden, pack_compressed_grid_sizes = self.pack_compressor.compress_latent(
                    raw_latent_batch
                )

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            pack_compressed_hidden=pack_compressed_hidden,
            pack_compressed_grid_sizes=pack_compressed_grid_sizes,
            pack_history_mid_latents=pack_history_mid_latents,
            pack_history_mid_hidden=pack_history_mid_hidden,
            pack_history_mid_grid_sizes=pack_history_mid_grid_sizes,
            pack_source_latent=raw_latent_batch,
        )

        def create_custom_forward(module):
            def custom_forward(
                x_inp,
                compressed_hidden_inp,
                compressed_grid_sizes_inp,
                history_mid_latents_inp,
                history_mid_hidden_inp,
                history_mid_grid_sizes_inp,
                *inputs,
                **inner_kwargs,
            ):
                call_kwargs = dict(inner_kwargs)
                call_kwargs.pop("pack_compressed_hidden", None)
                call_kwargs.pop("pack_compressed_grid_sizes", None)
                call_kwargs.pop("pack_history_mid_latents", None)
                call_kwargs.pop("pack_history_mid_hidden", None)
                call_kwargs.pop("pack_history_mid_grid_sizes", None)
                x_out = module(
                    x_inp,
                    *inputs,
                    pack_compressed_hidden=compressed_hidden_inp,
                    pack_compressed_grid_sizes=compressed_grid_sizes_inp,
                    pack_history_mid_latents=history_mid_latents_inp,
                    pack_history_mid_hidden=history_mid_hidden_inp,
                    pack_history_mid_grid_sizes=history_mid_grid_sizes_inp,
                    **call_kwargs,
                )
                return x_out
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            use_block_checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
            block_crossattn_cache = None if torch.is_grad_enabled() else crossattn_cache[block_index]
            block_forward = create_custom_forward(block)
            if use_block_checkpoint:
                checkpoint_kv_cache = kv_cache[block_index]
                if isinstance(checkpoint_kv_cache, PackLayerCacheState):
                    checkpoint_kv_cache = _snapshot_pack_layer_cache(checkpoint_kv_cache)
                checkpoint_kwargs = dict(kwargs)
                checkpoint_kwargs.pop("pack_compressed_hidden", None)
                checkpoint_kwargs.pop("pack_compressed_grid_sizes", None)
                checkpoint_kwargs.pop("pack_history_mid_latents", None)
                checkpoint_kwargs.pop("pack_history_mid_hidden", None)
                checkpoint_kwargs.pop("pack_history_mid_grid_sizes", None)
                checkpoint_kwargs.update(
                    {
                        "kv_cache": checkpoint_kv_cache,
                        "crossattn_cache": block_crossattn_cache,
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    x,
                    pack_compressed_hidden,
                    pack_compressed_grid_sizes,
                    pack_history_mid_latents,
                    pack_history_mid_hidden,
                    pack_history_mid_grid_sizes,
                    **checkpoint_kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": block_crossattn_cache,
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block_forward(
                    x,
                    pack_compressed_hidden,
                    pack_compressed_grid_sizes,
                    pack_history_mid_latents,
                    pack_history_mid_hidden,
                    pack_history_mid_grid_sizes,
                    **kwargs,
                )

        if (
            kv_cache is not None
            and self.pack_cfg.enable
            and self.pack_cfg.compress_mode == "hr_spatial"
            and not torch.is_grad_enabled()
        ):
            current_start_frame = current_start // int(torch.prod(grid_sizes[0][1:]).item())
            self._commit_source_latent_block(raw_latent_batch, current_start_frame)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs,
    ):
        use_internal_pack_kv_cache = kwargs.pop("pack_use_internal_kv_cache", False)
        if self.pack_cfg.enable and use_internal_pack_kv_cache:
            return self._forward_inference(*args, **kwargs)
        return super().forward(*args, **kwargs)


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
    precomputed_mid: CompressedKVBlock | None = None,
    source_latent: torch.Tensor | None = None,
) -> None:
    """Update the current block in-place across denoising steps, or append a new block."""

    tail = _get_tail_fullres_block(state)
    if tail is not None and tail.meta.start_frame == start_frame:
        updated_precomputed_mid = precomputed_mid if precomputed_mid is not None else tail.precomputed_mid
        updated = FullResKVBlock(
            k=roped_key,
            v=value,
            meta=tail.meta,
            precomputed_mid=updated_precomputed_mid,
        )
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
        precomputed_mid=precomputed_mid,
        source_latent=source_latent,
    )


def _get_tail_fullres_block(state: PackLayerCacheState) -> FullResKVBlock | None:
    if state.recent_blocks:
        return state.recent_blocks[-1]
    if state.sink_blocks:
        return state.sink_blocks[-1]
    return None


def _snapshot_pack_layer_cache(state: PackLayerCacheState) -> PackLayerCacheState:
    cached_selected_mid_indices = state.cached_selected_mid_indices
    if cached_selected_mid_indices is not None:
        cached_selected_mid_indices = cached_selected_mid_indices.clone()

    return PackLayerCacheState(
        cfg=state.cfg,
        sink_blocks=list(state.sink_blocks),
        recent_blocks=deque(state.recent_blocks),
        mid_blocks=deque(state.mid_blocks),
        next_block_id=state.next_block_id,
        total_committed_blocks=state.total_committed_blocks,
        batch_size=state.batch_size,
        cached_selected_mid_indices=cached_selected_mid_indices,
        cached_selection_for_start_frame=state.cached_selection_for_start_frame,
    )


def _history_state_without_current_block(
    state: PackLayerCacheState,
    current_start_frame: int,
) -> PackLayerCacheState:
    sink_blocks = list(state.sink_blocks)
    recent_blocks = list(state.recent_blocks)

    if recent_blocks and recent_blocks[-1].meta.start_frame == current_start_frame:
        recent_blocks = recent_blocks[:-1]
    elif sink_blocks and sink_blocks[-1].meta.start_frame == current_start_frame:
        sink_blocks = sink_blocks[:-1]

    return PackLayerCacheState(
        cfg=state.cfg,
        sink_blocks=sink_blocks,
        recent_blocks=deque(recent_blocks),
        mid_blocks=deque(state.mid_blocks),
        next_block_id=state.next_block_id,
        total_committed_blocks=state.total_committed_blocks,
        batch_size=state.batch_size,
        cached_selected_mid_indices=None,
        cached_selection_for_start_frame=None,
    )


def _state_with_mid_blocks(
    state: PackLayerCacheState,
    mid_blocks: list[CompressedKVBlock],
) -> PackLayerCacheState:
    return PackLayerCacheState(
        cfg=state.cfg,
        sink_blocks=list(state.sink_blocks),
        recent_blocks=deque(state.recent_blocks),
        mid_blocks=deque(mid_blocks),
        next_block_id=state.next_block_id,
        total_committed_blocks=state.total_committed_blocks,
        batch_size=state.batch_size,
        cached_selected_mid_indices=state.cached_selected_mid_indices,
        cached_selection_for_start_frame=state.cached_selection_for_start_frame,
    )


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
