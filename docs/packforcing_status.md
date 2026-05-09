# PackForcing Status

## Scope

This repository currently contains a minimal PackForcing inference branch built
on top of the Causal-Forcing codebase. The goal is to validate cache packing,
RoPE correction, and simple mid-selection heuristics before moving on to a
fuller reproduction path.

This is not yet a paper-complete PackForcing implementation.

## What Exists

- A separate PackForcing cache backend in
  [wan/modules/pack_cache.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/wan/modules/pack_cache.py)
  with:
  - `sink / mid / recent` block groups
  - block-level mid bank storage
  - block commit, eviction, selection, and attention-view assembly
  - placeholder compressors:
    - `identity`
    - `token_avg_pool`

- A separate PackForcing model path in
  [wan/modules/causal_model_packforcing.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/wan/modules/causal_model_packforcing.py)
  with:
  - `CausalWanSelfAttentionPackForcing`
  - `CausalWanAttentionBlockPackForcing`
  - `CausalWanModelPackForcing`

- A PackForcing inference entrypoint in
  [inference_packforcing.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/inference_packforcing.py)
  and a PackForcing wrapper in
  [utils/wan_wrapper.py](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/utils/wan_wrapper.py).

- An initial DMD training integration path:
  - the generator can now switch to the PackForcing wrapper from config
  - `SelfForcingTrainingPipeline` now uses the unified KV cache build/reset API
  - a starter config is provided in
    [configs/packforcing_dmd_chunkwise.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise.yaml)
  - a minimal one-step smoke config is provided in
    [configs/packforcing_dmd_chunkwise_smoke.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_smoke.yaml)
  - an 8-GPU 10-step validation config is provided in
    [configs/packforcing_dmd_chunkwise_8gpu_10step.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_8gpu_10step.yaml)
  - a formal 8-GPU launcher is provided in
    [scripts/train_packforcing_dmd_8gpu.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/train_packforcing_dmd_8gpu.sh)

## RoPE Status

- Packed-history RoPE correction is implemented.
- The current packed timeline uses an anchor-latest, backward-packing scheme.
- Current-block query and key stay on absolute time.
- History keys are shifted to packed time with a temporal complex phase delta.

The current phase-shift implementation assumes cached keys are already RoPE'd
in absolute time before any token-linear compression. This is valid for:

- `identity`
- `token_avg_pool`

It is not guaranteed to stay valid for future nonlinear or learned
compressors.

## Mid Selection Status

The current repository intentionally keeps mid selection simple. There are now
two supported debugging heuristics:

- `recency`
  - always prefers the mid blocks closest to `recent`
  - this is the current default because it is more stable in our 20s tests

- `query_score`
  - the original rough heuristic
  - scores each mid block with a simple query-summary and block-summary dot
    product

The selection mode is now a formal config instead of temporary hardcoded
behavior.

Relevant knobs:

- `pack_mid_select_topk_blocks`
- `pack_mid_selection_mode`

CLI example:

```bash
/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python inference_packforcing.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --checkpoint_path ckpt/causal_forcing_ckpt/causal_forcing.pt \
  --data_path prompts/packforcing_smoke_1prompt.txt \
  --output_folder output/tmp_packforcing \
  --num_output_frames 81 \
  --pack_enable \
  --pack_compress_mode identity \
  --pack_mid_select_topk_blocks 2 \
  --pack_mid_selection_mode recency \
  --pack_enable_rope_adjustment
```

## Key Findings So Far

- Without packed-history RoPE correction, identity-cache videos show large
  chunk-level jumps.
- After RoPE correction, those jumps are reduced and the result no longer snaps
  back toward the earliest frames.
- With dynamic `query_score` selection, chunk-to-chunk mid-set changes still
  introduce visible flicker.
- With `recency` selection, the 20s identity test becomes visibly more stable.

This strongly suggests that current instability is driven more by frequent
mid-selection switching than by the basic Pack cache structure.

## Preserved Reference Outputs

The main preserved PackForcing visual record is:

- [output/packforcing_record_rope_mid_selection_20s](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/output/packforcing_record_rope_mid_selection_20s)

This directory contains a compact 20s comparison showing:

- no RoPE correction
- RoPE correction with dynamic mid selection
- RoPE correction with `recency` mid selection

The DeepForcing comparison directory that is still kept is:

- [output/deepforcing_compare_60s](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/output/deepforcing_compare_60s)

## Known Gaps

- Only the DMD `SelfForcingTrainingPipeline` path has been wired for PackForcing so far
- No paper-complete mid-selection algorithm
- No learned PackForcing compressor
- No paper-complete long-video evaluation harness

## Training Smoke

A minimal PackForcing DMD training smoke now passes with the current codebase.

Validated command:

```bash
/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=1 train.py \
  --config_path configs/packforcing_dmd_chunkwise_smoke.yaml \
  --logdir logs/packforcing_dmd_chunkwise_smoke \
  --disable-wandb \
  --no_visualize \
  --no_save
```

Notes:

- the smoke config runs `max_train_steps: 1`
- `gradient_checkpointing` is enabled
- `text_encoder_cpu_offload: true`
- `num_workers: 0`
- the run uses the existing local conda env at
  `/beijing-c/workspace/hxj/miniconda3/envs/packforcing`

This smoke is intended to validate PackForcing training-path wiring and
backward execution, not to benchmark full training throughput.

## 8-GPU Validation

An 8-GPU FSDP PackForcing DMD validation run now passes for 10 steps.

Validated launcher:

```bash
./scripts/train_packforcing_dmd_8gpu_10step.sh
```

Current validation setup:

- uses `hybrid_full` sharding on 8x A100-80GB
- keeps the current repository's `3f21l` training scaffold
- uses `pack_sink_blocks: 2`
- uses `pack_recent_blocks: 1`
- uses `pack_mid_select_topk_blocks: 16`
- uses `pack_mid_selection_mode: recency`
- uses `pack_compress_mode: identity`
- points to the legacy prompt corpus at
  `/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial-legacy/prompts/vidprom_filtered_extended.txt`

This validates that the PackForcing training path is now runnable under
multi-GPU FSDP, not just single-GPU smoke.

## Current Training Default

The current DMD training default is intentionally conservative:

- compressor: `identity`
- mid selection: `recency`
- top-k mid blocks: `16`
- sink blocks: `2`

Under the current 5s `3f21l` training window, this means the training-time
history is still very close to the original Causal-Forcing KV-cache behavior.
The code path is PackForcing, but the training distribution shift remains
small until a learned or lossy compressor is introduced.

## Working Decision

For now:

- keep the current compressor implementations as-is
- keep the current heuristic mid-selection implementations as-is
- use formal config switches rather than adding more hidden hardcoded behavior
