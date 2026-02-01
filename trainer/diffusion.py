import gc
import logging

from model import CausalDiffusion
from utils.dataset import cycle, LatentLMDBDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
       
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9) 
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder
            
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
           
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts) 
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        if self.delta_mean is not None:
            clean_latent += self.delta_mean.to(device=clean_latent.device, dtype=clean_latent.dtype)
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    # =========================
    # Parallel "eval": mean tensor of (pred_latent - clean_latent) on predicted segment
    # pred has num_output_frames frames, and pred[:num_init_frames] == initial_latent (guaranteed)
    # =========================
    @torch.no_grad()
    def _generate_latents(self, prompts, *, initial_latent=None, num_output_frames=21, num_init_frames=3):
        bsz = len(prompts)
        device = torch.device("cuda", torch.cuda.current_device())
        noise_T = num_output_frames if initial_latent is None else (num_output_frames - num_init_frames)
        noise = torch.randn([bsz, noise_T, 16, 60, 104], device=device, dtype=self.dtype)

        latents = self.pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            return_video=False
        )
        return latents  # expected: [B, num_output_frames, 16, 60, 104]


    @torch.no_grad()
    def parallel_latent_algebraic_mean(self, index, *, num_output_frames=21, num_init_frames=3):
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = torch.device("cuda", torch.cuda.current_device())

        diff_sum = torch.zeros((num_output_frames, 16, 60, 104), device=device, dtype=torch.float32)
        cnt = torch.zeros((), device=device, dtype=torch.float32)

        prompt_index = int(index) * world + rank
        if prompt_index < len(self.dataset):
            sample = self.dataset[prompt_index]
            prompt = sample["prompts"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            
            clean = sample["clean_latent"].to(device=device, dtype=self.dtype)  # [>=21,16,60,104]
            clean = clean[:num_output_frames]
            print(f'clean.shape is {clean.shape}')
            init_latent = None
            
            if self.given_first_chunk:
                init_latent = clean[:num_init_frames].unsqueeze(0)

            latents = self._generate_latents(
                [prompt],
                initial_latent=init_latent,
                num_output_frames=num_output_frames,
                num_init_frames=num_init_frames
            )
            pred = latents[0].to(torch.float32)[:num_output_frames]             # [21,16,60,104]
            ref  = clean.to(torch.float32)                                      # [21,16,60,104]

            diff_sum.copy_(pred - ref)
            cnt.fill_(1.0)

            if hasattr(self.model, "vae") and hasattr(self.model.vae, "model") and hasattr(self.model.vae.model, "clear_cache"):
                self.model.vae.model.clear_cache()

        dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        return diff_sum / (cnt + 1e-12)



    def train(self):

        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # ---- parallel eval (chunk_size=1) ----
            if self.eval_interval and (self.step % self.eval_interval == 0):
                # all ranks must execute this block
                was_train_gen = self.model.generator.training
                was_train_txt = self.model.text_encoder.training
                self.model.generator.eval()
                self.model.text_encoder.eval()

                world = dist.get_world_size()
                # num_global = ceil(len(dataset)/world)
                num_global = (len(self.dataset) + world - 1) // world

                if num_global > 0:
                    for i in range(self.rtf_single_gpu_batch):
                        if dist.get_rank() == 0:
                            idx_t = torch.randint(0, num_global, (1,), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
                        else:
                            idx_t = torch.zeros((1,), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
                        dist.broadcast(idx_t, src=0)
                        index = int(idx_t.item())
                        
                        if i == 0:
                            diff_mean = self.parallel_latent_algebraic_mean(
                                index=index,
                                num_output_frames=self.eval_frames,
                                num_init_frames=self.eval_init
                            )
                        else :
                            diff_mean += self.parallel_latent_algebraic_mean(
                                index=index,
                                num_output_frames=self.eval_frames,
                                num_init_frames=self.eval_init
                            )
                            
                        diff_mean = diff_mean/self.rtf_single_gpu_batch
                        
                    if self.delta_mean is not None:
                        self.delta_mean = self.rtf_ema_ratio * self.delta_mean + (1 - self.rtf_ema_ratio) * diff_mean
                    else:
                        self.delta_mean = diff_mean

                        
                    val = self.delta_mean.mean().item()

                    if self.is_main_process and (not self.disable_wandb):
                        wandb.log({"delta_mean": val}, step=self.step)


                if was_train_gen:
                    self.model.generator.train()
                if was_train_txt:
                    self.model.text_encoder.train()


            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
