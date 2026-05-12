# DeepForcing Debug Notes

## Scope

This note records the current DeepForcing baseline migration status in `PackForcing-Unofficial`.

The goal here was not to exactly reproduce the full official repository layout, but to:

- move `sink only` and `sink + mid` inference logic into the Causal-Forcing codebase
- run them on top of `causal_forcing.pt`
- check which parts stay visually stable and which parts break

These notes are empirical conclusions from the current migration and debugging process, not a formal proof of paper correctness.

## Current Baselines

### Sink only

Current `sink only` baseline is kept in:

- `configs/deepforcing_chunkwise_sinkonly.yaml`

Current setting:

- `local_attn_size = 21`
- `sink_size = 12`

This matches the original Causal-Forcing sink-style rolling behavior more closely and is the stable baseline we keep.

### Sink + mid

Current `sink + mid` baseline is kept in:

- `configs/deepforcing_chunkwise_sinkmid.yaml`

Current setting:

- `local_attn_size = 21`
- `sink_size = 10`
- `pc_enable = true`
- `budget = 16`
- `recent = 4`
- `pc_mid_rope_unification = false`
- first-PC `bootstrap` is now fixed in code and no longer configurable

Operationally, this is the current mainline DeepForcing baseline in this repo.

## Main Findings

### 1. Sink + mid is more stable than sink only

After migrating DeepForcing logic into the Causal-Forcing codebase and testing long-video inference, the overall qualitative observation is:

- `sink + mid` is better than `sink only`
- compared with `sink only`, `sink + mid` is much less likely to suffer from subject disappearance

This does not mean every sample is strictly better in every aspect, but the subject-preservation behavior is noticeably stronger.

### 2. Official-style sink + mid path does not apply extra mid RoPE unification by default

During debugging, it became clear that the usable `sink + mid` path should not force an additional "continuous" temporal RoPE remapping over the selected mid history tokens.

In other words:

- the migrated `sink + mid` path works in its repo-faithful form
- adding an extra mid-history RoPE unification fix was an experiment
- that experiment did not help on top of `causal_forcing.pt`

### 3. Adding mid RoPE unification causes visual collapse

When temporal RoPE correction / unification was explicitly applied to the selected mid-history tokens, the output quality became unstable and often collapsed badly.

Observed behavior:

- image quality explodes
- motion or structure becomes unreliable
- overall output is much worse than the default `sink + mid` path

Current conclusion:

- do **not** enable `pc_mid_rope_unification` in the mainline baseline

### 4. Bootstrap collapse was caused by a bug in the initial sink delta

The official DeepForcing repository effectively includes a bootstrap delta when entering the PC path for the first time.

The earlier migration on top of `causal_forcing.pt` looked unstable when bootstrap was enabled, but that turned out to be an implementation bug rather than a valid modeling conclusion.

Root cause:

- the bootstrap path mixed token-space and frame-space offsets when computing the initial sink RoPE shift
- `_rope_time_delta_mul_()` expects a frame delta
- the wrong unit conversion produced an excessively large temporal RoPE jump and corrupted the sink remap

Current conclusion:

- keep bootstrap enabled as part of the default PC / `sink + mid` path
- do **not** expose bootstrap as a runtime toggle in this repo anymore

## Current Practical Conclusion

For this repo, the stable DeepForcing baseline to keep is:

- `sink only`: original stable `12 + 9` style baseline
- `sink + mid`: `pc_enable=true`, with the bootstrap fix applied in code and **without** mid RoPE unification

So the current working recommendation is:

- keep `sink only` as the conservative baseline
- use `sink + mid` as the stronger baseline for long-video subject retention
- do not re-enable `mid RoPE unification`

## What To Avoid Repeating

The following experiments have already been tried and are currently considered failed directions for this codebase / checkpoint combination:

- shrinking `sink only` into a forced `10 + 6` baseline by directly changing `local_attn_size`
- enabling `pc_mid_rope_unification`
- mixing token-space and frame-space offsets in bootstrap sink RoPE remapping

These changes either broke visual quality directly or moved the model too far away from the stable regime of `causal_forcing.pt`.

## Status

As of now, the repo should be understood as:

- DeepForcing sink-only and sink+mid logic have been migrated into the Causal-Forcing codebase
- the useful part of the migration is the default `sink + mid` path with the bootstrap fix applied
- `sink + mid` empirically helps with subject retention
- extra mid RoPE correction currently makes results worse
