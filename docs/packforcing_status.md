# PackForcing Status

## Scope

This repository now has:

- a working PackForcing inference path
- a working DMD training path
- an HR-only trainable spatial compressor
- FSDP-compatible cache/state handling for PackForcing training

This is still not a paper-complete PackForcing reproduction. The current code
is an engineering baseline for training and long-context inference experiments.

## Current Implementation

- Cache backend:
  [wan/modules/pack_cache.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/wan/modules/pack_cache.py)
  - `sink / mid / recent` block groups
  - block commit, eviction, selection, and attention-view assembly
  - supported compressors:
    - `identity`
    - `token_avg_pool`
    - `hr_spatial`

- Model path:
  [wan/modules/causal_model_packforcing.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/wan/modules/causal_model_packforcing.py)
  - `CausalWanSelfAttentionPackForcing`
  - `CausalWanAttentionBlockPackForcing`
  - `CausalWanModelPackForcing`
  - train-time history-mid rebuild from stored source latents
  - explicit metadata alignment check between source-mid history and per-layer mid KV cache
  - internal Pack layer cache state for FSDP-safe cross-chunk reuse

- HR compressor:
  [wan/modules/pack_compressor.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/wan/modules/pack_compressor.py)
  - HR-only
  - no temporal compression
  - 3-stage spatial downsampling
  - token order preserved as `Time -> Height -> Width`

- Wrapper:
  [utils/wan_wrapper.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/utils/wan_wrapper.py)
  - PackForcing wrapper uses model-internal Pack KV cache state
  - external `kv_cache` is now an opaque `PackCacheHandle`, not a mutable layer-cache list
  - PackForcing sessions expose two cache modes:
    - `rollout_compatible`
    - `finalized_chunk_only`
  - `commit_kv_cache(...)` is the unified helper for cache-only history commits
  - legacy generator checkpoints can load while initializing new compressor weights

## RoPE And Mid Selection

- Packed-history RoPE correction is implemented.
- The current packed timeline uses anchor-latest backward packing.
- Supported mid-selection modes:
  - `recency`
  - `query_score`
- Current default for stability is:
  - `pack_mid_selection_mode: recency`
  - `pack_mid_select_topk_blocks: 16`

## HR Compressor Status

- Current learned compressor path is `hr_spatial` only.
- No LR branch is implemented.
- No temporal compression is used.
- For the current `3f21l` setup, each compressed block still spans 3 latent frames.
- Current compressed token count per block is:
  `3 x (60 / 8) x (104 / 8) = 273`

## Training Path

The DMD training path is wired through:

- [pipeline/self_forcing_training.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/pipeline/self_forcing_training.py)
- [trainer/distillation.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/trainer/distillation.py)

Important current behavior:

- the last temporal block no longer reruns an extra `no_grad` generator forward before backward
- compressor observability is kept through:
  - `pack_comp_grad`
  - `pack_comp_update`
  - `pack_tokens`
- for `hr_spatial`, no-grad rollout steps populate both:
  - internal per-layer Pack KV cache
  - model-internal source latent history
- the grad-enabled target step rebuilds history-mid compressed hidden from the
  stored source latents, then replaces the cached mid KV view with trainable
  per-layer mid KV tensors for attention

## Inference Cache Semantics

The current inference path now separates cache ownership, cache mode, and cache
commit semantics explicitly.

- the real PackForcing cross-chunk state lives inside the wrapped model
- the outer pipeline only holds an opaque `PackCacheHandle`
- pure inference uses `finalized_chunk_only` mode
- training rollout continues to use `rollout_compatible` mode

For pure inference, denoising-step forwards are now read-only with respect to
Pack history. History only advances on explicit commit calls.

The unified commit path is:

- [utils/wan_wrapper.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/utils/wan_wrapper.py)
  - `commit_kv_cache(...)`

For PackForcing, that helper routes to:

- `pack_cache_commit=True`
- `pack_cache_only=True`

This means cache-seed and finalized-chunk refresh calls still execute:

- patch embedding
- transformer blocks
- Pack KV / source-latent commit

But they now skip:

- diffusion head
- unpatchify
- `flow_pred -> pred_x0` reconstruction

So the final inference-only cache refresh is now a cache-maintenance path, not
a second full output-producing forward.

## Recent Bug Fixes

On 2026-05-08 we found and fixed a real HR PackForcing cache bug in the
same-block update path.

Root cause:

- `_build_precomputed_mid_block()` creates transient mid metadata with
  `source_block_id=None`
- the first commit path normalizes that metadata correctly
- but repeated denoising updates for the same temporal block used
  `_upsert_pack_cache_block()` to overwrite `tail.precomputed_mid` directly
  without re-normalizing metadata
- this could silently erase `source_block_id` and break the identity mapping
  between:
  - `pack_source_state.mid_blocks`
  - per-layer `kv_cache.mid_blocks`

Current fix:

- same-block `precomputed_mid` updates are now normalized against the block
  metadata before writing back into cache
- training now performs an explicit alignment check on
  `(source_block_id, start_frame, num_frames)` before rebuilding history-mid
  hidden

Impact:

- before the fix, training could have fallen back to fragile positional
  lockstep between the two state machines
- with the new assertion, any future divergence should fail fast instead of
  silently corrupting history-mid supervision

## FSDP Lesson Learned

The main training bug we resolved was not in the HR compressor itself.

Under FSDP, it is unsafe to rely on:

- passing a complex mutable cache object through `forward`
- mutating it inside the wrapped module
- expecting the outer pipeline to keep seeing those mutations across chunks

In the old design, Pack layer KV cache lived outside the model and was passed as
`forward` input. Rollout appeared to update cache state, but target chunks did
not reliably see those updates under FSDP, so mid history stayed effectively
empty and the HR compressor was never consumed on the grad-enabled step.

The current fix is:

- Pack source history remains model-internal
- Pack layer KV cache is also model-internal
- cross-chunk PackForcing state is no longer implemented as an external mutable
  object whose reuse depends on implicit side effects across the FSDP boundary
- the external pipeline handle is now intentionally opaque, so callers no
  longer depend on model-internal layer-cache structure

## Configs To Keep

The retained PackForcing-specific configs are:

- [configs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2.yaml)
  - current main HR training recipe
- [configs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2.yaml)
  - cheaper 4-GPU regression / reproduction recipe
- [configs/packforcing_dmd_chunkwise_hr_strict_compare_fsdp.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_strict_compare_fsdp.yaml)
- [configs/packforcing_dmd_chunkwise_hr_strict_compare_nofsdp.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_strict_compare_nofsdp.yaml)
  - strict FSDP vs no-FSDP comparison

Older smoke-only YAMLs were transient debugging artifacts and have been removed
from `configs/`.

The retained HR scripts are:

- [scripts/train_packforcing_dmd_hr_8gpu.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/train_packforcing_dmd_hr_8gpu.sh)
- [scripts/compare_packforcing_fsdp_nofsdp.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/compare_packforcing_fsdp_nofsdp.sh)

## Validation Policy

Future core changes should be checked with strict compare first.

The intended rule is:

- strict compare is the first gate for cache / FSDP / HR-compressor changes
- the 4-GPU `grad_accum=2` recipe is the practical cheap training regression

Use:

```bash
./scripts/compare_packforcing_fsdp_nofsdp.sh
```

for any change that touches:

- cache/state handling
- FSDP wrapping
- gradient checkpointing interaction
- training-time history-mid rebuild
- HR compressor execution path

Latest strict-compare result on 2026-05-08:

- FSDP: `pack_comp_grad=0.029998`, `pack_comp_update=0.004000`, `pack_tokens=273`
- no-FSDP: `pack_comp_grad=0.020177`, `pack_comp_update=0.001191`, `pack_tokens=273`

Current pass criterion is:

- both branches finish the run
- both branches report non-zero `pack_comp_grad`
- both branches report non-zero `pack_comp_update`
- `pack_tokens` matches

Exact numeric equality is not currently required for this smoke.

Latest multi-GPU HR smoke results on 2026-05-08:

- `grad_accum=2`, `1 step`
  - log:
    [logs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2_smoke_rerun2/stdout.log](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/logs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2_smoke_rerun2/stdout.log)
  - result:
    - finished successfully
    - `pack_comp_grad=0.010710`
    - `pack_comp_update=0.003978`
    - `pack_tokens=273.0`

- `grad_accum=2`, `5 step`
  - log:
    [logs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2_5step_smoke_rerun/stdout.log](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/logs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2_5step_smoke_rerun/stdout.log)
  - result:
    - finished successfully
    - `pack_comp_grad=0.002420`
    - `pack_comp_update=0.003810`
    - `pack_tokens=273.0`

Latest cache-only inference / regression checks on 2026-05-08:

- single-card PackForcing inference regression with cache-only commit path passed
  - output:
    [a calm fox walking through a snowy forest.mp4](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/output/packforcing_inference_commit_helper_smoke/a%20calm%20fox%20walking%20through%20a%20snowy%20forest.mp4)
  - result:
    - finished successfully after switching both initial cache seed and finalized chunk refresh to `commit_kv_cache(...)`

- 4-GPU HR training regression after cache-only inference-path cleanup passed
  - config family:
    [configs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_4gpu_gradacc2.yaml)
  - result:
    - finished successfully
    - `g_loss=0.143202`
    - `pack_comp_grad=0.009126`
    - `pack_comp_update=0.003947`
    - `pack_tokens=273.0`

Current practical conclusion:

- the internal Pack KV cache path is working for the current DMD +
  SelfForcing + FSDP training setup
- the HR compressor is being exercised on the grad-enabled step
- both the 1-step and 5-step `grad_accum=2` smokes completed without:
  - cache metadata mismatch
  - FSDP collective hangs
  - the old `clip_grad_norm_()` false hang

## Stage Summary

What this branch has already achieved:

- PackForcing-style KV cache semantics are implemented on top of the causal
  forcing model path
- the cache is now model-internal and FSDP-safe instead of being a fragile
  external mutable object
- a trainable `hr_spatial` compressor is wired into the backward path
- inference has explicit cache ownership and explicit cache-commit semantics

What the recent experiments suggest:

- `sink=2 / recent=1` tends to collapse toward a static-looking solution
- `sink=3 / recent=2` is closer to the paper-style recipe, but the model still
  drifts toward stasis by later training steps
- the current implementation is therefore functionally working, but not yet a
  faithful long-horizon reproduction

Most plausible mismatch sources, in priority order:

- the current path is still closer to `causal forcing` than paper-style
  `self forcing`
- mid selection is still simplified (`recency` / `query_score` + top-k), not a
  paper-complete selection rule
- RoPE packing/correction is engineering-consistent, but may not match the
  paper's exact long-horizon state evolution
- the current HR compressor is spatial-only and does not include the paper's
  full dual-branch / temporal-compression behavior

Working hypothesis:

- the model is learning a conservative low-motion fixed point that is stable
  under the current rollout and cache semantics, but does not extrapolate well
  beyond a few seconds
- extending the reachable RoPE range alone is likely not enough if the training
  rollout semantics remain mismatched

## Debugging Workflow

PackForcing debug runs should be launched in `tmux`, not in an ephemeral
foreground shell.

The standard rule is:

- check free GPUs first
- start the run in a dedicated `tmux` session
- pipe stdout/stderr through `tee` into a stable log file under `logs/`
- record the session name, config path, log path, attach command, and stop command

This avoids losing logs when an interactive session is interrupted and makes it
easy to inspect progress or stop a run safely.

## Dataset Note

PackForcing training and validation should use:

`/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial-legacy/prompts/vidprom_filtered_extended.txt`

for multi-GPU training smoke and validation.

The single-prompt file is still useful for strict compare and very small local
tests, but it is not suitable for multi-GPU smoke when
`DistributedSampler(..., drop_last=True)` is active.

## Current Defaults

The current maintained HR training default is:

- `pack_sink_blocks: 3`
- `pack_recent_blocks: 2`
- `pack_mid_select_topk_blocks: 16`
- `pack_mid_selection_mode: recency`
- `pack_compress_mode: hr_spatial`

The 8-GPU `sink=3 / recent=2` recipe is now the main maintained training path,
while the 4-GPU `grad_accum=2` recipe is kept as the cheaper regression path.

## Known Gaps

- Only the DMD `SelfForcingTrainingPipeline` path is wired for PackForcing
- No paper-complete mid-selection algorithm
- No LR compressor branch
- No paper-complete long-video evaluation harness
- Current long-horizon extrapolation still degrades toward static output after
  enough training, so the reproduction is not yet paper-faithful at scale
