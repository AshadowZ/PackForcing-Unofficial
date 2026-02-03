<div align="center">

# Causal Forcing
### Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation

<p align="center">
  <p align="center">
    <div>
    <a href="https://zhuhz22.github.io/" target="_blank">Hongzhou Zhu*</a><sup></sup>,
    <a href="https://gracezhao1997.github.io/" target="_blank">Min Zhao*</a><sup></sup> , 
    <a href="https://guandehe.github.io/" target="_blank">Guande He</a><sup></sup>, 
    <a href="https://scholar.google.com/citations?user=dxN1_X0AAAAJ&hl=en" target="_blank">Hang Su</a><sup></sup>,
    <a href="https://zhenxuan00.github.io/" target="_blank">Chongxuan Li</a><sup></sup> ,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml" target="_blank">Jun Zhu</a><sup></sup>
</div>
<div>
    <sup></sup>Tsinghua University & Shengshu & UT Austin
</div>


</div>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2602.02214">Paper</a> | <a href="https://thu-ml.github.io/CausalForcing.github.io">Website</a> | <a href="https://huggingface.co/zhuhz22/Causal-Forcing/tree/main">Models</a> | <a href="assets/wechat.jpg">WeChat</a> </h3>
</p>



-----


Causal Forcing significantly outperforms Self Forcing in **both visual quality and motion dynamics**, while keeping **the same training budget and inference efficiency**—enabling real-time, streaming video generation on a single RTX 4090.


-----



https://github.com/user-attachments/assets/310f0cfa-e1bb-496d-8941-87f77b3271c0



## Quick Start

> The inference environment is identical to Self Forcing, so you can migrate directly using our configs and model.


### Installation
```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```
### Download checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir wan_models/Wan2.1-T2V-14B
hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing framewise/causal_forcing.pt --local-dir checkpoints
```

### CLI Inference
> We open-source both the frame-wise and chunk-wise models; the former is a setting that Self Forcing has chosen not to release.

Frame-wise model (**higher dynamic degree and more expressive**):
```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise.yaml \
  --output_folder output/framewise \
  --checkpoint_path  checkpoints/framewise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --use_ema
    # Note: this frame-wise config not in Self Forcing; if using its framework, migrate this config too.
```

Chunk-wise model (**more stable**):
```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder output/chunkwise \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt
```



## Training
<img width="4944" height="2154" alt="overview" src="https://github.com/user-attachments/assets/df96fae3-cecc-4915-9a14-d1a5f326074e" />

<details>
<summary> Stage 1: Autoregressive Diffusion Training (Can skip by using our pretrained checkpoints. Click to expand.)</summary>

First download the dataset (we provide a 6K toy dataset here):
```bash
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset
python utils/merge_and_get_clean.py
```
> If the download gets stuck, Ctrl^C and then resume it.


Then train the AR-diffusion model:
- Framewise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_framewise.yaml \
    --logdir logs/ar_diffusion_framewise
  ```

- Chunkwise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_chunkwise.yaml \
    --logdir logs/ar_diffusion_chunkwise
  ```

> We recommend training no less than 2K steps, and more steps (e.g., 5~10K) will lead to better performance.


</details>


<details>
<summary> Stage 2: Causal ODE Initialization (Can skip by using our pretrained checkpoints. Click to expand.)</summary>

If you have skipped Stage 1, you need to download the pretrained models:
```bash
hf download zhuhz22/Causal-Forcing framewise/ar_diffusion.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt --local-dir checkpoints
```

In this stage, first generate ODE paired data:
```bash
# for the frame-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_framewise.py \
  --generator_ckpt checkpoints/framewise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_framewise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_framewise_latents \
  --lmdb_path dataset/ODE6KCausal_framewise

# for the chunk-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_chunkwise.py \
  --generator_ckpt checkpoints/chunkwise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_chunkwise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_chunkwise_latents \
  --lmdb_path dataset/ODE6KCausal_chunkwise
```

Or you can also directly download our prepared dataset (~300G):
```bash
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset
python utils/merge_lmdb.py
```
> If the download gets stuck, Ctrl^C and then resume it.


And then train ODE initialization models:
- Frame-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_framewise.yaml \
    --logdir logs/causal_ode_framewise
  ```
- Chunk-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_chunkwise.yaml \
    --logdir logs/causal_ode_chunkwise
  ```

> We recommend training no less than 1K steps, and more steps (e.g., 5~10K) will lead to better performance.

</details>

### Stage 3: DMD

> This stage is compatible with Self Forcing training, so you can migrate seamlessly by using our configs and checkpoints.

> Set your wandb configs before training.

First download the dataset:
```bash
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
If you have skipped Stage 2, you need to download the pretrained checkpoints:
```bash
hf download zhuhz22/Causal-Forcing framewise/causal_ode.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/causal_ode.pt --local-dir checkpoints
```

And then train DMD models:

- Frame-wise model:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_framewise.yaml \
    --logdir logs/causal_forcing_dmd_framewise
  ```
  > We recommend training 500 steps. More than 1K steps will reduce dynamic degree.


- Chunk-wise model:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_chunkwise.yaml \
    --logdir logs/causal_forcing_dmd_chunkwise
  ```
  > We recommend training 100~200 steps. More than 1K steps will reduce dynamic degree.

Such models are the final models used to generate videos.
## QA & Blog 
> currently in Chinese

See [here](https://zhuanlan.zhihu.com/p/2002114039493461457).

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## References
If you find the method useful, please cite
```
@misc{zhu2026causalforcingautoregressivediffusion,
      title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation}, 
      author={Hongzhou Zhu and Min Zhao and Guande He and Hang Su and Chongxuan Li and Jun Zhu},
      year={2026},
      eprint={2602.02214},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.02214}, 
}
```
