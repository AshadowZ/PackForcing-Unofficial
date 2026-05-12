from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)


def _rope_time_delta_mul_(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
    """Apply an in-place temporal RoPE shift to already-roped sink keys."""
    if delta_frames == 0:
        return

    _, _, _, d = k_chunk.shape
    assert d % 2 == 0
    c = d // 2
    time_c = c - 2 * (c // 3)
    height_c = c // 3
    width_c = c // 3
    freqs_t, _, _ = freqs.split([time_c, height_c, width_c], dim=1)

    shift = min(abs(int(delta_frames)), freqs_t.shape[0] - 1)
    multiplier = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    multiplier = multiplier.view(1, 1, 1, time_c)

    time_ri = k_chunk[..., : 2 * time_c]
    time_cx = torch.view_as_complex(
        time_ri.to(torch.float64).reshape(-1, time_c, 2)
    )
    time_cx = time_cx * multiplier.to(time_cx.dtype)
    time_ri_new = torch.view_as_real(time_cx).reshape(
        k_chunk.shape[0], k_chunk.shape[1], k_chunk.shape[2], time_c, 2
    ).flatten(-2)
    time_ri.copy_(time_ri_new.to(time_ri.dtype))


def _rope_time_delta_mul_per_token_(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: torch.Tensor) -> None:
    """Apply a per-token temporal RoPE shift in-place to already-roped mid keys."""
    if delta_frames.numel() == 0:
        return

    batch_size, token_count, _, d = k_chunk.shape
    if delta_frames.dim() == 1:
        delta_frames = delta_frames.unsqueeze(0).expand(batch_size, -1)
    elif delta_frames.dim() == 2 and delta_frames.shape[0] == 1 and batch_size != 1:
        delta_frames = delta_frames.expand(batch_size, -1)

    if delta_frames.shape != (batch_size, token_count):
        raise ValueError(
            f"delta_frames shape {tuple(delta_frames.shape)} does not match expected {(batch_size, token_count)}."
        )

    delta_frames = delta_frames.to(device=k_chunk.device, dtype=torch.long)
    if not torch.any(delta_frames != 0):
        return

    assert d % 2 == 0
    c = d // 2
    time_c = c - 2 * (c // 3)
    height_c = c // 3
    width_c = c // 3
    freqs_t, _, _ = freqs.split([time_c, height_c, width_c], dim=1)

    max_pos = freqs_t.shape[0] - 1
    delta_index = delta_frames.abs().clamp(max=max_pos)
    multiplier = freqs_t.index_select(0, delta_index.reshape(-1)).reshape(batch_size, token_count, time_c)
    if torch.any(delta_frames < 0):
        multiplier = torch.where((delta_frames < 0).unsqueeze(-1), torch.conj(multiplier), multiplier)
    multiplier = multiplier.unsqueeze(2)

    time_ri = k_chunk[..., : 2 * time_c]
    time_cx = torch.view_as_complex(
        time_ri.to(torch.float64).reshape(batch_size, token_count, k_chunk.shape[2], time_c, 2)
    )
    time_cx = time_cx * multiplier.to(time_cx.dtype)
    time_ri_new = torch.view_as_real(time_cx).reshape(
        batch_size, token_count, k_chunk.shape[2], time_c, 2
    ).flatten(-2)
    time_ri.copy_(time_ri_new.to(time_ri.dtype))



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class PCConfig:
    def __init__(self, enable=False, capacity=1560 * 15, window=1560 * 4,
                 fusion="sum", keep_sinks=True, topc_max_reuse=0,
                 mid_rope_unification=False, mid_rope_mode=None):
        self.enable = enable
        self.capacity = int(capacity)
        self.window = int(window)
        self.fusion = fusion
        self.keep_sinks = keep_sinks
        self.topc_max_reuse = max(0, int(topc_max_reuse))
        if mid_rope_mode is None:
            mid_rope_mode = "segment" if mid_rope_unification else "none"
        self.mid_rope_mode = str(mid_rope_mode)
        if self.mid_rope_mode not in {"none", "segment", "token"}:
            raise ValueError(
                f"Unsupported mid_rope_mode={self.mid_rope_mode!r}; expected 'none', 'segment', or 'token'."
            )
        self.mid_rope_unification = self.mid_rope_mode != "none"


def _mkv_update_win_q(kv_cache, new_win_q, window_tokens):
    new_win_q = new_win_q.detach()
    if "win_q" not in kv_cache or kv_cache["win_q"] is None:
        kv_cache["win_q"] = new_win_q
    else:
        kv_cache["win_q"] = torch.cat([kv_cache["win_q"], new_win_q], dim=1)
    if kv_cache["win_q"].shape[1] > window_tokens:
        kv_cache["win_q"] = kv_cache["win_q"][:, -window_tokens:]


def _mkv_select_indices(scores_fused, total_tokens, recent_tokens, sink_tokens, top_c,
                        device, topc_counts=None, topc_max_reuse=0):
    batch_size = scores_fused.shape[0]
    keep_lists = []
    protect_lists = []
    topc_selected_lists = []
    recent_idx = torch.arange(max(0, total_tokens - recent_tokens), total_tokens, device=device)
    sink_idx = (
        torch.arange(0, sink_tokens, device=device)
        if sink_tokens > 0 else torch.tensor([], device=device, dtype=torch.long)
    )

    old_end = max(0, total_tokens - recent_tokens)
    cand_start = min(sink_tokens, old_end)
    cand_len = max(0, old_end - cand_start)
    if cand_len == 0 or top_c <= 0:
        for _ in range(batch_size):
            keep_lists.append(torch.cat([sink_idx, recent_idx]).clone())
            protect_lists.append(torch.tensor([], device=device, dtype=torch.long))
            topc_selected_lists.append(torch.tensor([], device=device, dtype=torch.long))
        return keep_lists, protect_lists, topc_selected_lists

    candidate_idx = torch.arange(cand_start, old_end, device=device)
    k = min(int(top_c), cand_len)

    for batch_idx in range(batch_size):
        scores_b = scores_fused[batch_idx, cand_start:old_end]
        counts_b = None
        if topc_counts is not None:
            counts_b = topc_counts[batch_idx, cand_start:old_end]

        if counts_b is not None and topc_max_reuse > 0:
            valid_mask = counts_b < topc_max_reuse
            if not torch.any(valid_mask):
                selected_b = torch.tensor([], device=device, dtype=torch.long)
            else:
                allowed_scores = scores_b[valid_mask]
                allowed_idx = candidate_idx[valid_mask]
                k_eff = min(k, allowed_idx.numel())
                if k_eff > 0:
                    _, top_local = torch.topk(allowed_scores, k=k_eff, dim=0)
                    selected_b = torch.sort(allowed_idx[top_local])[0]
                else:
                    selected_b = torch.tensor([], device=device, dtype=torch.long)
        else:
            k_eff = min(k, candidate_idx.numel())
            if k_eff > 0:
                _, top_local = torch.topk(scores_b, k=k_eff, dim=0)
                selected_b = torch.sort(candidate_idx[top_local])[0]
            else:
                selected_b = torch.tensor([], device=device, dtype=torch.long)

        protect_b = torch.unique(selected_b, sorted=True)
        keep_b = torch.unique(torch.cat([sink_idx, protect_b, recent_idx]), sorted=True)
        keep_lists.append(keep_b)
        protect_lists.append(protect_b)
        topc_selected_lists.append(protect_b)

    return keep_lists, protect_lists, topc_selected_lists


def _mkv_prune_cache(
    kv_cache,
    keep_lists,
    protect_lists,
    sink_tokens,
    topc_selected_lists=None,
    topc_max_reuse=0,
    source_k=None,
    source_v=None,
    source_abs=None,
    source_topc_counts=None,
):
    key_dst = kv_cache["k"]
    val_dst = kv_cache["v"]
    key_src = source_k if source_k is not None else key_dst
    val_src = source_v if source_v is not None else val_dst
    batch_size, cache_capacity, num_heads, head_dim = key_dst.shape
    device = key_dst.device
    max_keep = max([len(idx) for idx in keep_lists]) if keep_lists else 0

    new_key = torch.zeros((batch_size, max_keep, num_heads, head_dim), dtype=key_dst.dtype, device=device)
    new_val = torch.zeros((batch_size, max_keep, num_heads, head_dim), dtype=val_dst.dtype, device=device)
    new_mask = torch.zeros((batch_size, max_keep), dtype=torch.bool, device=device)
    abs_idx = kv_cache.get("abs_frame_idx", None)
    abs_src = source_abs if source_abs is not None else abs_idx
    topc_counts = kv_cache.get("topc_select_counts", None)
    counts_src = source_topc_counts if source_topc_counts is not None else topc_counts

    new_counts = None
    if topc_counts is not None:
        new_counts = torch.zeros_like(topc_counts)
    if abs_idx is not None:
        new_abs = torch.full((batch_size, max_keep), -1, dtype=abs_idx.dtype, device=device)

    for batch_idx in range(batch_size):
        idx = keep_lists[batch_idx]
        keep_len = len(idx)
        if keep_len == 0:
            continue
        new_key[batch_idx, :keep_len] = key_src[batch_idx, idx]
        new_val[batch_idx, :keep_len] = val_src[batch_idx, idx]
        if abs_idx is not None and abs_src is not None:
            new_abs[batch_idx, :keep_len] = abs_src[batch_idx, idx]
        if new_counts is not None and counts_src is not None:
            new_counts[batch_idx, :keep_len] = counts_src[batch_idx, idx]
            if topc_max_reuse > 0 and topc_selected_lists is not None:
                selected = topc_selected_lists[batch_idx]
                if selected.numel() > 0:
                    selected_set = set(selected.tolist())
                    for pos_new, pos_old in enumerate(idx.tolist()):
                        if pos_old in selected_set:
                            new_counts[batch_idx, pos_new] += 1
        protect = protect_lists[batch_idx]
        if protect.numel() > 0:
            keep_list = idx.tolist()
            protect_set = set(protect.tolist())
            protect_positions = [pos for pos, old_idx in enumerate(keep_list) if old_idx in protect_set]
            if protect_positions:
                new_mask[batch_idx, torch.tensor(protect_positions, device=device)] = True

    key_dst.zero_()
    val_dst.zero_()
    if max_keep > 0:
        key_dst[:, :max_keep] = new_key[:, :max_keep]
        val_dst[:, :max_keep] = new_val[:, :max_keep]

    if abs_idx is not None:
        abs_idx.fill_(-1)
        if max_keep > 0:
            abs_idx[:, :max_keep] = new_abs[:, :max_keep]

    if new_counts is not None:
        topc_counts.zero_()
        topc_counts[:, :max_keep] = new_counts[:, :max_keep]

    kv_cache["local_end_index"].fill_(max_keep)
    kv_cache["protected_mask"] = torch.zeros((batch_size, cache_capacity), dtype=torch.bool, device=device)
    kv_cache["protected_mask"][:, :max_keep] = new_mask
    kv_cache["protected_len"] = kv_cache["protected_mask"].sum(dim=1).to(torch.long)
    kv_cache["protected_len_max"] = (
        kv_cache["protected_len"].max()
        if kv_cache["protected_len"].numel() > 0
        else torch.tensor(0, dtype=torch.long, device=device)
    )


class CausalWanSelfAttentionDeepForcing(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 PC: PCConfig | None = None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.PC = PC or PCConfig(enable=False)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            sink_tokens_v = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            prev_local_end = kv_cache["local_end_index"].item()
            batch_cache = kv_cache["k"].shape[0]
            if "abs_frame_idx" not in kv_cache:
                kv_cache["abs_frame_idx"] = torch.full(
                    (kv_cache["k"].shape[0], kv_cache_size),
                    -1,
                    dtype=torch.long,
                    device=kv_cache["k"].device,
                )
            abs_frame_idx = kv_cache["abs_frame_idx"]
            if "topc_select_counts" not in kv_cache:
                kv_cache["topc_select_counts"] = torch.zeros(
                    (kv_cache["k"].shape[0], kv_cache_size),
                    dtype=torch.long,
                    device=kv_cache["k"].device,
                )
            topc_select_counts = kv_cache["topc_select_counts"]
            token_offsets_new = torch.arange(num_new_tokens, device=kv_cache["k"].device)
            new_abs_frames = current_start_frame + torch.div(token_offsets_new, frame_seqlen, rounding_mode='floor')
            new_abs_frames = new_abs_frames.to(abs_frame_idx.dtype)
            rolled_condition = self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + prev_local_end > kv_cache_size
            )
            available_slots = kv_cache_size - prev_local_end
            need_evict = False
            new_tokens_integrated = False

            if self.PC.enable:
                _mkv_update_win_q(kv_cache, roped_query, window_tokens=self.PC.window)

            if rolled_condition:
                if self.PC.enable and available_slots < num_new_tokens:
                    cached_len = prev_local_end
                    key_existing = kv_cache["k"][:, :cached_len]
                    val_existing = kv_cache["v"][:, :cached_len]
                    if cached_len > 0:
                        key_aug = torch.cat([key_existing, roped_key], dim=1)
                        val_aug = torch.cat([val_existing, v], dim=1)
                        new_abs_tile = new_abs_frames.unsqueeze(0).expand(batch_cache, -1)
                        abs_existing = abs_frame_idx[:, :cached_len]
                        abs_aug = torch.cat([abs_existing, new_abs_tile], dim=1)
                        zeros_new = torch.zeros(
                            (batch_cache, num_new_tokens),
                            dtype=topc_select_counts.dtype,
                            device=topc_select_counts.device,
                        )
                        counts_existing = topc_select_counts[:, :cached_len]
                        counts_aug = torch.cat([counts_existing, zeros_new], dim=1)
                    else:
                        key_aug = roped_key
                        val_aug = v
                        abs_aug = new_abs_frames.unsqueeze(0).expand(batch_cache, -1)
                        counts_aug = torch.zeros(
                            (batch_cache, num_new_tokens),
                            dtype=topc_select_counts.dtype,
                            device=topc_select_counts.device,
                        )

                    cached_len_aug = cached_len + num_new_tokens
                    win_q = kv_cache.get("win_q", None)
                    recent_eff = min(win_q.shape[1], cached_len_aug) if win_q is not None else 0
                    if recent_eff > 0 and cached_len_aug > 0:
                        recent_q = win_q[:, -recent_eff:]
                        k_flat = key_aug.reshape(key_aug.shape[0], cached_len_aug, -1)
                        k_trans = k_flat.transpose(1, 2).contiguous()
                        scale = 1.0 / (math.sqrt(self.head_dim) * self.num_heads)
                        if self.PC.fusion == "sum":
                            q_sum_flat = recent_q.reshape(recent_q.shape[0], recent_eff, -1).sum(dim=1, keepdim=True)
                            fused = torch.bmm(q_sum_flat, k_trans).squeeze(1).to(torch.float32)
                            fused.mul_(scale)
                        else:
                            q_flat = recent_q.reshape(recent_q.shape[0], recent_eff, -1)
                            fused = torch.full(
                                (q_flat.shape[0], cached_len_aug),
                                -float("inf"),
                                device=key_aug.device,
                                dtype=torch.float32
                            )
                            step = max(1, min(256, recent_eff))
                            for start in range(0, recent_eff, step):
                                end = min(recent_eff, start + step)
                                q_chunk = q_flat[:, start:end]
                                scores_chunk = torch.matmul(q_chunk, k_trans).to(torch.float32)
                                scores_chunk.mul_(scale)
                                chunk_max = scores_chunk.amax(dim=1)
                                fused = torch.maximum(fused, chunk_max)

                        forced_sink = min(max(sink_tokens, sink_tokens_v), cached_len_aug) if self.PC.keep_sinks else 0
                        fused[:, max(0, cached_len_aug - recent_eff):] = -float("inf")
                        if forced_sink > 0:
                            fused[:, :forced_sink] = -float("inf")
                        total_cap = min(int(self.PC.capacity), kv_cache_size)
                        total_cap = max(total_cap, forced_sink + recent_eff)
                        total_cap = min(total_cap, cached_len_aug)
                        top_c = max(0, total_cap - forced_sink - recent_eff)
                        keep_lists, protect_lists, topc_selected = _mkv_select_indices(
                            fused, total_tokens=cached_len_aug, recent_tokens=recent_eff,
                            sink_tokens=forced_sink, top_c=top_c, device=key_aug.device,
                            topc_counts=counts_aug, topc_max_reuse=self.PC.topc_max_reuse
                        )
                        _mkv_prune_cache(
                            kv_cache,
                            keep_lists,
                            protect_lists,
                            forced_sink,
                            topc_selected_lists=topc_selected,
                            topc_max_reuse=self.PC.topc_max_reuse,
                            source_k=key_aug,
                            source_v=val_aug,
                            source_abs=abs_aug,
                            source_topc_counts=counts_aug,
                        )
                        prev_local_end = kv_cache["local_end_index"].item()
                        available_slots = kv_cache_size - prev_local_end
                        new_tokens_integrated = True

                if not new_tokens_integrated and available_slots < num_new_tokens:
                    need_evict = True

            if need_evict:
                num_evicted_tokens = num_new_tokens + prev_local_end - kv_cache_size
                num_rolled_tokens = max(0, prev_local_end - num_evicted_tokens - sink_tokens)
                num_rolled_tokens_v = max(0, prev_local_end - num_evicted_tokens - sink_tokens_v)
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                )
                kv_cache["v"][:, sink_tokens_v:sink_tokens_v + num_rolled_tokens_v] = (
                    kv_cache["v"][:, sink_tokens_v + num_evicted_tokens:sink_tokens_v + num_evicted_tokens + num_rolled_tokens_v].clone()
                )
                abs_frame_idx[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    abs_frame_idx[
                        :,
                        sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens
                    ].clone()
                )
                if sink_tokens + num_rolled_tokens < prev_local_end:
                    abs_frame_idx[:, sink_tokens + num_rolled_tokens:prev_local_end] = -1
                local_end_index = prev_local_end + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                topc_select_counts[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    topc_select_counts[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                )
                if sink_tokens + num_rolled_tokens < prev_local_end:
                    topc_select_counts[:, sink_tokens + num_rolled_tokens:prev_local_end] = 0
                topc_select_counts[:, local_start_index:local_end_index] = 0
            elif not new_tokens_integrated:
                local_end_index = prev_local_end + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                topc_select_counts[:, local_start_index:local_end_index] = 0
            else:
                local_end_index = kv_cache["local_end_index"].item()
                local_start_index = max(0, local_end_index - num_new_tokens)

            if not new_tokens_integrated:
                insert_len = local_end_index - local_start_index
                if insert_len > 0:
                    token_offsets = torch.arange(insert_len, device=kv_cache["k"].device)
                    abs_frames = current_start_frame + torch.div(token_offsets, frame_seqlen, rounding_mode='floor')
                    abs_frame_idx[:, local_start_index:local_end_index] = abs_frames
            elif kv_cache["k"].shape[1] > local_end_index:
                abs_frame_idx[:, local_end_index:] = -1

            window_start = max(0, local_end_index - self.max_attention_size)
            key_win = kv_cache["k"][:, window_start:local_end_index]
            val_win = kv_cache["v"][:, window_start:local_end_index]
            if rolled_condition:
                sink_len_tokens = min(local_end_index, sink_tokens)

                if self.PC.enable and self.PC.mid_rope_mode != "none":
                    recent_keep_tokens = min(
                        self.PC.window,
                        max(0, local_end_index - sink_len_tokens)
                    )
                    mid_start = sink_len_tokens
                    recent_start = max(mid_start, local_end_index - recent_keep_tokens)
                    mid_end = recent_start
                    recent_end = local_end_index

                    mid_len_tokens = max(0, mid_end - mid_start)
                    recent_len_tokens = max(0, recent_end - recent_start)
                    sink_len_frames = math.ceil(sink_len_tokens / frame_seqlen) if sink_len_tokens > 0 else 0
                    mid_len_frames = math.ceil(mid_len_tokens / frame_seqlen) if mid_len_tokens > 0 else 0
                    recent_len_frames = math.ceil(recent_len_tokens / frame_seqlen) if recent_len_tokens > 0 else 0

                    desired_recent_abs_start = current_start_frame - recent_len_frames
                    desired_mid_abs_start = desired_recent_abs_start - mid_len_frames
                    desired_sink_abs_start = desired_mid_abs_start - sink_len_frames

                    if self.sink_size > 0 and sink_len_tokens > 0:
                        if "sink_base_abs_start_frame" not in kv_cache:
                            kv_cache["sink_base_abs_start_frame"] = torch.tensor(
                                desired_sink_abs_start, device=kv_cache["k"].device
                            )
                            local_window_frames = self.max_attention_size // frame_seqlen
                            delta_sink = local_window_frames - (self.PC.capacity // frame_seqlen)
                        else:
                            delta_sink = int(
                                desired_sink_abs_start - kv_cache["sink_base_abs_start_frame"].item()
                            )
                        if delta_sink != 0:
                            _rope_time_delta_mul_(kv_cache["k"][:, :sink_len_tokens], freqs, delta_sink)
                            kv_cache["sink_base_abs_start_frame"].fill_(desired_sink_abs_start)
                        sink_offsets = torch.arange(sink_len_tokens, device=kv_cache["k"].device)
                        sink_frames = desired_sink_abs_start + torch.div(
                            sink_offsets, frame_seqlen, rounding_mode='floor'
                        )
                        kv_cache["abs_frame_idx"][:, :sink_len_tokens] = sink_frames

                    if mid_len_tokens > 0:
                        mid_offsets = torch.arange(mid_len_tokens, device=kv_cache["k"].device)
                        desired_mid_frames = desired_mid_abs_start + torch.div(
                            mid_offsets, frame_seqlen, rounding_mode='floor'
                        )
                        if self.PC.mid_rope_mode == "token":
                            desired_mid_frames_batch = desired_mid_frames.unsqueeze(0).expand(batch_cache, -1)
                            current_mid_frames = abs_frame_idx[:, mid_start:mid_end]
                            delta_mid = torch.where(
                                current_mid_frames >= 0,
                                desired_mid_frames_batch - current_mid_frames,
                                torch.zeros_like(desired_mid_frames_batch),
                            )
                            if torch.any(delta_mid != 0):
                                _rope_time_delta_mul_per_token_(
                                    kv_cache["k"][:, mid_start:mid_end],
                                    freqs,
                                    delta_mid,
                                )
                        else:
                            if "topc_base_abs_start_frame" not in kv_cache:
                                kv_cache["topc_base_abs_start_frame"] = torch.tensor(
                                    desired_mid_abs_start, device=kv_cache["k"].device
                                )
                            delta_mid = int(
                                desired_mid_abs_start - kv_cache["topc_base_abs_start_frame"].item()
                            )
                            if delta_mid != 0:
                                _rope_time_delta_mul_(kv_cache["k"][:, mid_start:mid_end], freqs, delta_mid)
                                kv_cache["topc_base_abs_start_frame"].fill_(desired_mid_abs_start)
                        kv_cache["abs_frame_idx"][:, mid_start:mid_end] = desired_mid_frames.unsqueeze(0).expand(batch_cache, -1)

                    # After PC pruning, cache layout is already [sink][mid][recent].
                    key_win = kv_cache["k"][:, :local_end_index]
                    val_win = kv_cache["v"][:, :local_end_index]
                else:
                    tail_end = local_end_index
                    tail_start = max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens)
                    tail_len_tokens = tail_end - tail_start
                    tail_len_frames = tail_len_tokens // frame_seqlen
                    sink_len_frames = sink_len_tokens // frame_seqlen
                    tail_start_abs_frame = current_start_frame - tail_len_frames
                    desired_sink_abs_start = tail_start_abs_frame - sink_len_frames

                    if self.sink_size > 0 and sink_len_tokens > 0:
                        if "sink_base_abs_start_frame" not in kv_cache:
                            kv_cache["sink_base_abs_start_frame"] = torch.tensor(
                                desired_sink_abs_start, device=kv_cache["k"].device
                            )
                            if self.PC.enable:
                                local_window_frames = self.max_attention_size // frame_seqlen
                                delta = local_window_frames - (self.PC.capacity // frame_seqlen)
                            else:
                                delta = 0
                        else:
                            delta = int(
                                desired_sink_abs_start - kv_cache["sink_base_abs_start_frame"].item()
                            )
                        if delta != 0:
                            _rope_time_delta_mul_(kv_cache["k"][:, :sink_len_tokens], freqs, delta)
                            kv_cache["sink_base_abs_start_frame"].fill_(desired_sink_abs_start)
                        sink_offsets = torch.arange(sink_len_tokens, device=kv_cache["k"].device)
                        sink_frames = desired_sink_abs_start + torch.div(
                            sink_offsets, frame_seqlen, rounding_mode='floor'
                        )
                        kv_cache["abs_frame_idx"][:, :sink_len_tokens] = sink_frames

                        key_win = torch.cat([
                            kv_cache["k"][:, :sink_len_tokens],
                            kv_cache["k"][:, tail_start:tail_end]
                        ], dim=1)
                        val_win = torch.cat([
                            kv_cache["v"][:, :sink_len_tokens],
                            kv_cache["v"][:, tail_start:tail_end]
                        ], dim=1)
                    else:
                        key_win = kv_cache["k"][:, tail_start:tail_end]
                        val_win = kv_cache["v"][:, tail_start:tail_end]

                    if self.PC.enable:
                        top_start = sink_tokens
                        top_end = tail_start
                        top_len_tokens = max(0, top_end - top_start)
                        if top_len_tokens > 0:
                            top_len_frames = math.ceil(top_len_tokens / frame_seqlen)
                            desired_top_abs_start = tail_start_abs_frame - top_len_frames
                            if "topc_base_abs_start_frame" not in kv_cache:
                                kv_cache["topc_base_abs_start_frame"] = torch.tensor(
                                    desired_top_abs_start, device=kv_cache["k"].device
                                )
                            delta_top = int(
                                desired_top_abs_start - kv_cache["topc_base_abs_start_frame"].item()
                            )
                            if delta_top != 0:
                                _rope_time_delta_mul_(kv_cache["k"][:, top_start:top_end], freqs, delta_top)
                                kv_cache["topc_base_abs_start_frame"].fill_(desired_top_abs_start)
                            top_offsets = torch.arange(top_len_tokens, device=kv_cache["k"].device)
                            top_frames = desired_top_abs_start + torch.div(
                                top_offsets, frame_seqlen, rounding_mode='floor'
                            )
                            kv_cache["abs_frame_idx"][:, top_start:top_end] = top_frames

            x = attention(roped_query, key_win, val_win)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = self.o(x)
        return x


class CausalWanAttentionBlockDeepForcing(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 PC: PCConfig | None = None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttentionDeepForcing(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, PC=PC
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
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
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModelDeepForcing(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['CausalWanAttentionBlockDeepForcing']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
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
                 PC_enable: bool = False,
                 PC_capacity: int = 1560 * 16,
                 PC_window: int = 1560 * 4,
                 PC_fusion: str = "sum",
                 PC_keep_sinks: bool = True,
                 PC_topc_max_reuse: int = 7,
                 PC_mid_rope_unification: bool = False,
                 PC_mid_rope_mode=None):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.PC_cfg = PCConfig(
            enable=PC_enable,
            capacity=PC_capacity,
            window=PC_window,
            fusion=PC_fusion,
            keep_sinks=PC_keep_sinks,
            topc_max_reuse=PC_topc_max_reuse,
            mid_rope_unification=PC_mid_rope_unification,
            mid_rope_mode=PC_mid_rope_mode,
        )

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlockDeepForcing(cross_attn_type, dim, ffn_dim, num_heads,
                                               local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                               PC=self.PC_cfg)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

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
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None: # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
