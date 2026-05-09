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

## Configs To Keep

The retained HR configs are:

- [configs/packforcing_dmd_chunkwise_hr.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr.yaml)
  - formal HR training config
- [configs/packforcing_dmd_chunkwise_hr_8gpu_smoke.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_8gpu_smoke.yaml)
  - 8-GPU one-step validation
- [configs/packforcing_dmd_chunkwise_hr_8gpu_10step.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_8gpu_10step.yaml)
  - 8-GPU 10-step validation
- [configs/packforcing_dmd_chunkwise_hr_smoke.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_smoke.yaml)
  - single-card functional smoke
- [configs/packforcing_dmd_chunkwise_hr_strict_compare_fsdp.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_strict_compare_fsdp.yaml)
- [configs/packforcing_dmd_chunkwise_hr_strict_compare_nofsdp.yaml](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/configs/packforcing_dmd_chunkwise_hr_strict_compare_nofsdp.yaml)
  - strict FSDP vs no-FSDP comparison

The retained HR scripts are:

- [scripts/train_packforcing_dmd_hr_8gpu.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/train_packforcing_dmd_hr_8gpu.sh)
- [scripts/train_packforcing_dmd_hr_8gpu_10step.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/train_packforcing_dmd_hr_8gpu_10step.sh)
- [scripts/compare_packforcing_fsdp_nofsdp.sh](/beijing-c/workspace/hxj/a-glj-ws/AR-Video/PackForcing-Unofficial/scripts/compare_packforcing_fsdp_nofsdp.sh)

## Validation Policy

Future core changes should be checked with strict compare first.

The intended rule is:

- functional smoke confirms the code still runs
- strict compare confirms FSDP and no-FSDP still behave similarly

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

The current training defaults remain conservative:

- `pack_sink_blocks: 2`
- `pack_recent_blocks: 1`
- `pack_mid_select_topk_blocks: 16`
- `pack_mid_selection_mode: recency`
- `pack_compress_mode: identity` for conservative baseline training

The HR path is implemented and trainable, but the repository still keeps the
identity path available as the lowest-risk baseline.

## Known Gaps

- Only the DMD `SelfForcingTrainingPipeline` path is wired for PackForcing
- No paper-complete mid-selection algorithm
- No LR compressor branch
- No paper-complete long-video evaluation harness
