import argparse
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.wan_wrapper import (
    WanDiffusionWrapperPackForcing,
    WanTextEncoder,
    WanVAEWrapper,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of latent frames to generate")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--sink_size", type=int, default=None, help="Override sink size in frames")
parser.add_argument("--local_attn_size", type=int, default=None, help="Override local attention window in frames")
parser.add_argument("--pack_enable", action="store_true", help="Enable the PackForcing cache path")
parser.add_argument("--pack_sink_blocks", type=int, default=None, help="Override PackForcing sink block count")
parser.add_argument("--pack_recent_blocks", type=int, default=None, help="Override PackForcing recent block count")
parser.add_argument("--pack_mid_bank_capacity_blocks", type=int, default=None, help="Override PackForcing mid bank capacity")
parser.add_argument("--pack_mid_select_topk_blocks", type=int, default=None, help="Override PackForcing top-k mid block selection")
parser.add_argument(
    "--pack_mid_selection_mode",
    type=str,
    choices=["recency", "query_score"],
    default=None,
    help="Override PackForcing mid block selection heuristic",
)
parser.add_argument("--pack_compress_mode", type=str, default=None, help="Override PackForcing compressor mode")
parser.add_argument("--pack_compressed_tokens_per_block", type=int, default=None, help="Override compressed tokens per mid block")
parser.add_argument("--pack_enable_rope_adjustment", action="store_true", help="Enable PackForcing packed-history RoPE adjustment")
args = parser.parse_args()


if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda")
    local_rank = 0

set_seed(args.seed)

print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

if not hasattr(config, "denoising_step_list"):
    raise ValueError("PackForcing smoke-test inference only supports the causal few-step inference path.")

if not hasattr(config, "model_kwargs") or config.model_kwargs is None:
    config.model_kwargs = OmegaConf.create({})

if args.sink_size is not None:
    config.model_kwargs["sink_size"] = int(args.sink_size)
if args.local_attn_size is not None:
    config.model_kwargs["local_attn_size"] = int(args.local_attn_size)
if args.pack_enable:
    config.model_kwargs["pack_enable"] = True
if args.pack_sink_blocks is not None:
    config.model_kwargs["pack_sink_blocks"] = int(args.pack_sink_blocks)
if args.pack_recent_blocks is not None:
    config.model_kwargs["pack_recent_blocks"] = int(args.pack_recent_blocks)
if args.pack_mid_bank_capacity_blocks is not None:
    config.model_kwargs["pack_mid_bank_capacity_blocks"] = int(args.pack_mid_bank_capacity_blocks)
if args.pack_mid_select_topk_blocks is not None:
    config.model_kwargs["pack_mid_select_topk_blocks"] = int(args.pack_mid_select_topk_blocks)
if args.pack_mid_selection_mode is not None:
    config.model_kwargs["pack_mid_selection_mode"] = str(args.pack_mid_selection_mode)
if args.pack_compress_mode is not None:
    config.model_kwargs["pack_compress_mode"] = str(args.pack_compress_mode)
if args.pack_compressed_tokens_per_block is not None:
    config.model_kwargs["pack_compressed_tokens_per_block"] = int(args.pack_compressed_tokens_per_block)
if args.pack_enable_rope_adjustment:
    config.model_kwargs["pack_enable_rope_adjustment"] = True

generator = WanDiffusionWrapperPackForcing(
    **dict(config.model_kwargs),
    is_causal=True,
)
pipeline = CausalInferencePipeline(
    config,
    device=device,
    generator=generator,
    text_encoder=WanTextEncoder(),
    vae=WanVAEWrapper(),
)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if args.use_ema else "generator"
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
print(f"Number of prompts: {len(dataset)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


for _, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    del idx

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]
    else:
        raise TypeError(f"Unsupported batch_data type: {type(batch_data)}")

    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        prompt = batch["prompts"][0]
        output_path = os.path.join(args.output_folder, f"{prompt[:100]}.mp4")
        if os.path.exists(output_path):
            print("Video has been generated. Pass!")
            continue

        image = batch["image"].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt]
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        prompt = batch["prompts"][0]
        output_path = os.path.join(args.output_folder, f"{prompt[:100]}.mp4")
        if os.path.exists(output_path):
            print("Video has been generated. Pass!")
            continue

        extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
        prompts = [extended_prompt] if extended_prompt is not None else [prompt]
        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
    )
    current_video = video.permute(0, 1, 3, 4, 2).cpu()
    video = 255.0 * current_video
    del latents

    pipeline.vae.model.clear_cache()
    write_video(output_path, video[0], fps=16)
