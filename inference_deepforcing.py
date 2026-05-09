import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.wan_wrapper import (
    WanDiffusionWrapperDeepForcing,
    WanTextEncoder,
    WanVAEWrapper,
)

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--sink_size", type=int, default=None, help="Override sink size in frames")
parser.add_argument("--local_attn_size", type=int, default=None, help="Override local attention window in frames")
parser.add_argument("--pc_enable", action="store_true", help="Enable DeepForcing sink+mid / participative compression")
parser.add_argument("--budget", type=int, default=None, help="Override total retained budget in latent frames")
parser.add_argument("--recent", type=int, default=None, help="Override recent window in latent frames")
parser.add_argument("--pc_mid_rope_unification", action="store_true", help="Enable experimental mid/top-c temporal RoPE unification")
parser.add_argument("--pc_bootstrap_delta", action="store_true", help="Enable the original DeepForcing first-PC sink bootstrap delta")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

if not hasattr(config, "denoising_step_list"):
    raise ValueError("DeepForcing baseline only supports the causal few-step inference path.")

if not hasattr(config, "model_kwargs") or config.model_kwargs is None:
    config.model_kwargs = OmegaConf.create({})
if args.sink_size is not None:
    config.model_kwargs["sink_size"] = int(args.sink_size)
if args.local_attn_size is not None:
    config.model_kwargs["local_attn_size"] = int(args.local_attn_size)
if args.pc_enable:
    config.model_kwargs["pc_enable"] = True
if args.budget is not None:
    config.model_kwargs["budget"] = int(args.budget)
if args.recent is not None:
    config.model_kwargs["recent"] = int(args.recent)
if args.pc_mid_rope_unification:
    config.model_kwargs["pc_mid_rope_unification"] = True
if args.pc_bootstrap_delta:
    config.model_kwargs["pc_bootstrap_delta"] = True

generator = WanDiffusionWrapperDeepForcing(
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
    key = 'generator_ema' if args.use_ema else 'generator'
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


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    write_video(output_path, video[0], fps=16)

       
