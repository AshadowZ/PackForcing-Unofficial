# AGENTS.md

Instructions for coding agents working in this repository.

## Project Intent

This repository starts from a clean `Causal-Forcing` base and is used to
validate the core PackForcing ideas with the smallest reasonable amount of new
code.

The focus is narrow:

- validate PackForcing-style bounded cache behavior on top of Causal-Forcing
- validate compression, context selection, and RoPE-related ideas
- keep method validation separate from large-scale recipe reproduction

Unless explicitly requested, this repository is not for rebuilding the old
`4f20l` reproduction track or reintroducing its surrounding training
complexity.

## Working Principles

- Keep changes small, reviewable, and reversible.
- Preserve original Causal-Forcing behavior when PackForcing is disabled.
- Keep the baseline runnable at all times.
- Add new behavior behind explicit config switches before changing defaults.
- Prefer inference-first validation before introducing training-path complexity.
- Prefer adding small helper modules over overloading a single large file.
- Do not silently couple unrelated experiments together.
- Do not remove upstream code paths unless the user explicitly asks for cleanup.
- Write all test videos and temporary validation outputs under the repository
  root `output/` directory, not under `long_video/`.
- Treat long-video behavior as an ablation problem.
- Avoid large coupled changes that make failures hard to attribute.

## Debug Notes

- Read `docs/deepforcing_debug.md` before changing the current DeepForcing
  sink-only / sink-mid baseline behavior.
- Preferred runtime environment for this repository is the existing conda env
  at `/beijing-c/workspace/hxj/miniconda3/envs/packforcing`.
- When running local smoke tests or inference scripts, prefer invoking the
  interpreter directly as
  `/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python ...`
  instead of creating or mutating another environment.
- Treat this as a hard default for all local validation in this repo:
  `train.py`, `inference*.py`, `demo.py`, and `torchrun` smoke tests should run
  from the `packforcing` environment unless the user explicitly asks for a
  different one.
- Do not fall back to the base environment for PackForcing validation. Missing
  packages or ABI mismatches there can produce unrelated failures and waste
  debugging time.

## Commit Format

Use short Conventional Commit-style messages:

```text
<type>(<scope>): <imperative summary>
```

Scope is optional. Common types:
`feat`, `fix`, `docs`, `refactor`, `test`, `config`, `chore`.

Examples:

```text
feat(cache): add sink-mid-recent cache skeleton
fix(inference): preserve baseline path when cache is disabled
docs: clarify repository scope
```

Keep the summary imperative, focused, and without a trailing period.
