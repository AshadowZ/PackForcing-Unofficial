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
