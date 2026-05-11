# PackForcing-Unofficial

This repository is an unofficial implementation of
[PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference](https://arxiv.org/abs/2603.25730). It is a non-strict reproduction of PackForcing's core ideas on top of the [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) `chunkwise 3f21l` codebase, rather than a paper-faithful reimplementation.

The repo was built out of curiosity about how PackForcing can train and infer
with such different RoPE ranges without immediately collapsing. I am also a
beginner, so this was a casual paper-reproduction exercise to learn by doing.

## What Is Here

- [x] Ported training-free Deep Forcing as a baseline.
- [x] Implemented PackForcing KV cache semantics, including three-stage memory, top-k mid selection, and RoPE correction.
- [x] Implemented a trainable HR Compressor.
- [ ] Training-free LR Compressor.

## Differences From the Paper

- Built on Causal Forcing rather than Self Forcing. I do not expect this to be
  the main blocker, and it may even behave better.
- Uses `chunkwise 3f21l` rather than `4f20l`, mainly to avoid redoing the full
  ODE-initialization pipeline.
- The HR Compressor removes temporal compression because the base setup is
  `3f21l`. My current intuition is that this mostly lowers the compression
  ratio.
- Reuses the Causal Forcing training dataset. The PackForcing paper seems to
  use a somewhat different dataset setup.

## Current Status

I ported training-free Deep Forcing as a baseline. This baseline is
`causal_forcing.pt + Deep Forcing`, and I will place the sink-only / sink+mid
comparison videos here later.

I trained two 8-GPU runs for 3k steps with no gradient accumulation and global
batch size 8: one with `sink=2 chunk + recent=1 chunk`, and one with
`sink=3 chunk + recent=2 chunk`. As training goes on, the dynamic degree gets
lower, which matches the trend already mentioned in the Causal Forcing README.
The `sink=2 / recent=1` run stays low-motion almost throughout, while
`sink=3 / recent=2` looks better around step 1000.

The videos above were checked with inference `top-k=3/1`, not the paper-style
`top-k=16`. I also kept the inference-time RoPE range close to the training
range there. When I switch back to `top-k=16`, the 20-second inference result
blows up much more easily, and I still do not know which mismatch is the main
cause.

## Usage

### Environment Setup

Use the same environment and data setup as
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing). The default
training initialization uses `ckpt/causal_forcing_ckpt/causal_ode.pt`, so you
need to download that checkpoint into the expected local path first.

### CLI Inference

```bash
python inference_packforcing.py \
  --config_path configs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2.yaml \
  --checkpoint_path <your_packforcing_checkpoint.pt> \
  --data_path prompts/<your_prompt_file.txt> \
  --output_folder output/<your_run_name> \
  --use_ema
```

### Training

The default 8-GPU training recipe is:

```bash
./scripts/train_packforcing_dmd_hr_8gpu.sh
```

It defaults to `configs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2.yaml`,
and the training behavior is broadly consistent with the paper (maybe).

## Discussion

1. While building this repo, I found that something like Deep Forcing can
   already achieve surprisingly good infinite continuation by just treating the
   first few frames as sink and adjusting the RoPE behavior. In practice, the
   extra trained PackForcing-style model does not yet look dramatically better
   than the training-free baseline here, which makes me wonder what the real
   gain of training is under the current setup.

2. I also found that Causal Forcing and Self Forcing are not as interchangeable
   as I first assumed. For example, when I directly add Deep Forcing bootstrap
   tricks on top of Causal Forcing, the video can collapse badly instead of
   acting like a clean drop-in replacement. Next time I try to reproduce a
   paper, I should probably stay closer to the original setup, otherwise bugs
   and implementation mismatches become hard to disentangle in ablations.

## Acknowledgements

This codebase builds on ideas and components from:
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[Deep Forcing](https://github.com/cvlab-kaist/DeepForcing),
[PackForcing](https://github.com/ShandaAI/PackForcing).

## Citation

If you use this fork, please cite the upstream Causal Forcing paper:

```bibtex
@article{zhu2026causal,
  title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
  author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2602.02214},
  year={2026}
}
```
