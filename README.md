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

### Training-Free Deep Forcing Baseline

Baseline: `causal_forcing.pt + Deep Forcing`. In the comparison videos below,
the left side uses `sink + recent`, while the right side uses
`sink + mid top-k + recent` as the KV cache.

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/4a52b253-2e0e-4bee-9d73-f220ed517a95" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/bf9545d8-0bfa-4775-8e9a-cc8e638a4b35" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/a82e2bca-681c-4ba4-b204-8f5fd85ecb79" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/17d32436-b2f9-4973-8fe5-5aea383f97a9" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
  </tr>
</table>

### PackForcing Reproduction Result

I also tried a more paper-like setup: `sink=3`, `recent=2`, `mid top-k=16`
(chunk units), trained for 3k steps on 8 GPUs with global batch size 8 and no
gradient accumulation.

There is indeed no large-scale color drift or full collapse as generation goes
on. However, the aerial-view sample still breaks, there is a visible jump
around 5-10 seconds, the frames remain noisy throughout, and gray blocks
sometimes appear in the top-left corner.

Next I will probably run a few more experiments to narrow down where these
issues come from. If they can all be resolved, the result should be fairly
close to the PackForcing paper.

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/574418e5-1782-4ace-917d-2f16d36b5190" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/ef02915a-9eb8-451e-9efa-3cf836e92695" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/507e4ffe-7ce3-4ca4-a359-76b653eabdb0" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/727fa6aa-e33f-4d6a-92f6-c3f5d8f8505b" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
  </tr>
</table>

### Debugging Notes

I also ran a few extra checks. For example, when I reduce inference-time
`top-k` selection from `16` to `1`, the train/infer RoPE range stays aligned,
and the visible jumps disappear, which makes me suspect the jump issue is
related to RoPE. (This is also the phenomenon observed in [LoL](https://arxiv.org/abs/2601.16914) and [RIFLEx](https://arxiv.org/pdf/2502.15894): abrupt changes and repetition. Could they be related?) I also tracked how the frames change across training steps:
the videos tend to become more static as training goes on, while the gray
blocks get gradually suppressed. My guess is that the gray-block artifact shows
up because the HR Compressor is still not fully trained.

In the comparison below, the left video uses `top-k=16`, while the right video
uses `top-k=1`. Both are 20-second inference results.

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/1957318a-24b0-4d5f-8c1f-a3d75d7d2fe9" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/348bdf1f-a4b2-405b-ba05-c97f7f0621fd" controls muted playsinline preload="metadata" width="100%"></video>
    </td>
  </tr>
</table>

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

1. As this reproduction moved forward, I found that training-free methods such
   as [Deep Forcing](https://github.com/cvlab-kaist/DeepForcing) and
   [Infinite-Forcing](https://github.com/SOTAMak1r/Infinite-Forcing) can
   already produce reasonably watchable infinite continuation. As long as the
   first few frames are kept and the RoPE behavior is corrected, the video can
   stay visually coherent without the colors collapsing over time.

2. I also found that Self Forcing itself tends to produce relatively
   low-motion videos. More generally, these trained infinite-continuation
   methods seem to prefer lowering the motion further, because once the motion
   gets too large, the generation can more easily drift outside the memory
   patterns seen during training and collapse.

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
