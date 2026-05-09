import gc
import logging
from contextlib import ExitStack, nullcontext
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.disable_generator_fsdp_wrap = getattr(config, "disable_generator_fsdp_wrap", False)

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.grad_accum = int(getattr(config, "grad_accum", 1))
        self.grad_accum_use_no_sync = bool(getattr(config, "grad_accum_use_no_sync", True))
        if self.grad_accum <= 0:
            raise ValueError(f"grad_accum must be positive, got {self.grad_accum}.")
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            expected_total_batch_size = config.batch_size * self.world_size * self.grad_accum
            if total_batch_size != expected_total_batch_size:
                raise ValueError(
                    "total_batch_size must equal batch_size * world_size * grad_accum, "
                    f"got total_batch_size={total_batch_size}, batch_size={config.batch_size}, "
                    f"world_size={self.world_size}, grad_accum={self.grad_accum}."
                )

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
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        if self.disable_generator_fsdp_wrap:
            self.model.generator = self.model.generator.to(
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.model.generator = fsdp_wrap(
                self.model.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                cpu_offload=False,
            )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
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

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "num_workers", 8))

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
            allow_missing_pack_compressor = (
                getattr(getattr(self.model.generator, "module", self.model.generator), "model", None) is not None
                and getattr(
                    getattr(getattr(self.model.generator, "module", self.model.generator), "model", None),
                    "pack_cfg",
                    None,
                ) is not None
                and getattr(
                    getattr(getattr(self.model.generator, "module", self.model.generator), "model", None).pack_cfg,
                    "compress_mode",
                    None,
                )
                == "hr_spatial"
            )
            incompatible = self.model.generator.load_state_dict(
                state_dict, strict=not allow_missing_pack_compressor
            )
            if allow_missing_pack_compressor:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [key for key in missing if "pack_compressor" not in key]
                if disallowed_missing or unexpected:
                    raise RuntimeError(
                        "Unexpected generator checkpoint mismatch when loading PackForcing hr_spatial "
                        f"weights. Missing keys: {disallowed_missing}; unexpected keys: {unexpected}"
                    )
                if missing:
                    print(
                        "Initialized new PackForcing compressor weights from scratch:",
                        ", ".join(missing),
                    )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self._packforcing_compressor_grad_sq_accum = None
        self._packforcing_compressor_grad_hooks = []
        self._register_packforcing_compressor_grad_hooks()

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
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

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(
        self,
        batch,
        train_generator,
        clean_latent=None,
        backward_scale: float = 1.0,
        compute_grad_norms: bool = True,
    ):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

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

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            (generator_loss * backward_scale).backward()

            generator_log_dict.update({"generator_loss": generator_loss})
            if compute_grad_norms:
                packforcing_compressor_grad_norm = self._packforcing_compressor_grad_norm()
                generator_grad_norm = self._clip_grad_norm(
                    self.model.generator,
                    self.max_grad_norm_generator,
                )
                generator_log_dict["generator_grad_norm"] = generator_grad_norm
                if packforcing_compressor_grad_norm is not None:
                    generator_log_dict["packforcing_compressor_grad_norm"] = packforcing_compressor_grad_norm
            packforcing_compressed_tokens_per_block = self._packforcing_compressed_tokens_per_block()
            if packforcing_compressed_tokens_per_block is not None:
                generator_log_dict["packforcing_compressed_tokens_per_block"] = (
                    packforcing_compressed_tokens_per_block
                )

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        (critic_loss * backward_scale).backward()

        critic_log_dict.update({"critic_loss": critic_loss})
        if compute_grad_norms:
            critic_grad_norm = self._clip_grad_norm(
                self.model.fake_score,
                self.max_grad_norm_critic,
            )
            critic_log_dict["critic_grad_norm"] = critic_grad_norm

        return critic_log_dict


    def train(self):
        start_step = self.step
        max_train_steps = getattr(self.config, "max_train_steps", None)
       
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                self._reset_packforcing_compressor_grad_stats()
                packforcing_compressor_param_snapshot = self._snapshot_packforcing_compressor_params()
                generator_log_dicts = []
                for accum_step in range(self.grad_accum):
                    batch = next(self.dataloader)
                    sync_gradients = accum_step == self.grad_accum - 1
                    with self._no_sync_context(
                        [self.model.generator],
                        enabled=self.grad_accum_use_no_sync and not sync_gradients,
                    ):
                        generator_log_dicts.append(
                            self.fwdbwd_one_step(
                                batch,
                                True,
                                backward_scale=1.0 / self.grad_accum,
                                compute_grad_norms=sync_gradients,
                            )
                        )
                generator_log_dict = self._average_log_dicts(generator_log_dicts)

                self.generator_optimizer.step()
                packforcing_compressor_update_norm = self._packforcing_compressor_update_norm(
                    packforcing_compressor_param_snapshot
                )
                if packforcing_compressor_update_norm is not None:
                    generator_log_dict["packforcing_compressor_update_norm"] = (
                        packforcing_compressor_update_norm
                    )
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dicts = []
            for accum_step in range(self.grad_accum):
                batch = next(self.dataloader)
                sync_gradients = accum_step == self.grad_accum - 1
                with self._no_sync_context(
                    [self.model.fake_score],
                    enabled=self.grad_accum_use_no_sync and not sync_gradients,
                ):
                    critic_log_dicts.append(
                        self.fwdbwd_one_step(
                            batch,
                            False,
                            backward_scale=1.0 / self.grad_accum,
                            compute_grad_norms=sync_gradients,
                        )
                    )
            critic_log_dict = self._average_log_dicts(critic_log_dicts)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    if "packforcing_compressor_grad_norm" in generator_log_dict:
                        wandb_loss_dict["packforcing_compressor_grad_norm"] = (
                            generator_log_dict["packforcing_compressor_grad_norm"].mean().item()
                        )
                    if "packforcing_compressed_tokens_per_block" in generator_log_dict:
                        wandb_loss_dict["packforcing_compressed_tokens_per_block"] = (
                            generator_log_dict["packforcing_compressed_tokens_per_block"]
                        )
                    if "packforcing_compressor_update_norm" in generator_log_dict:
                        wandb_loss_dict["packforcing_compressor_update_norm"] = (
                            generator_log_dict["packforcing_compressor_update_norm"].mean().item()
                        )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                if TRAIN_GENERATOR:
                    summary = (
                        f"[step {self.step + 1}] "
                        f"g_loss={wandb_loss_dict['generator_loss']:.6f} "
                        f"g_grad={wandb_loss_dict['generator_grad_norm']:.6f} "
                        f"critic_loss={wandb_loss_dict['critic_loss']:.6f}"
                    )
                    if "packforcing_compressor_grad_norm" in wandb_loss_dict:
                        summary += (
                            f" pack_comp_grad={wandb_loss_dict['packforcing_compressor_grad_norm']:.6f}"
                        )
                    if "packforcing_compressed_tokens_per_block" in wandb_loss_dict:
                        summary += (
                            f" pack_tokens={wandb_loss_dict['packforcing_compressed_tokens_per_block']}"
                        )
                    if "packforcing_compressor_update_norm" in wandb_loss_dict:
                        summary += (
                            f" pack_comp_update={wandb_loss_dict['packforcing_compressor_update_norm']:.6f}"
                        )
                    print(summary)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if max_train_steps is not None and self.step >= max_train_steps:
                if self.is_main_process:
                    print(f"Reached max_train_steps={max_train_steps}, stopping training loop.")
                break

    def _packforcing_compressor_grad_norm(self):
        if self._packforcing_compressor_grad_hooks:
            hook_squared_norm = self._packforcing_compressor_grad_sq_accum
            if hook_squared_norm is None:
                hook_squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                hook_squared_norm = hook_squared_norm.detach().clone()
            hook_grad_seen = torch.tensor(
                1 if hook_squared_norm.item() > 0 else 0,
                device=hook_squared_norm.device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                dist.all_reduce(hook_squared_norm, op=dist.ReduceOp.SUM)
                dist.all_reduce(hook_grad_seen, op=dist.ReduceOp.SUM)
            if hook_grad_seen.item() > 0:
                return hook_squared_norm.sqrt()

        try:
            compressor = self._get_packforcing_compressor()
            if compressor is None:
                return None
        except Exception:
            return None

        squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        grad_seen = torch.zeros((), device=self.device, dtype=torch.long)
        for param in compressor.parameters():
            if param.grad is None:
                continue
            value = param.grad.detach().float().norm(2).pow(2)
            squared_norm = squared_norm + value
            grad_seen.fill_(1)
        if dist.is_initialized():
            dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(grad_seen, op=dist.ReduceOp.SUM)
        if grad_seen.item() == 0:
            return None
        return squared_norm.sqrt().to(self.device)

    def _no_sync_context(self, modules, enabled: bool):
        if not enabled:
            return nullcontext()

        stack = ExitStack()
        for module in modules:
            if hasattr(module, "no_sync"):
                stack.enter_context(module.no_sync())
        return stack

    def _average_log_dicts(self, log_dicts):
        if not log_dicts:
            return {}

        averaged = {}
        all_keys = set().union(*(log_dict.keys() for log_dict in log_dicts))
        for key in all_keys:
            values = [log_dict[key] for log_dict in log_dicts if key in log_dict]
            if not values:
                continue
            first = values[0]
            if torch.is_tensor(first):
                averaged[key] = torch.stack([value.detach() for value in values]).mean(dim=0)
            elif isinstance(first, (int, float)):
                averaged[key] = sum(values) / len(values)
            else:
                averaged[key] = first
        return averaged

    def _unwrap_fsdp_like_module(self, module):
        current = module
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "module"):
                current = current.module
                continue
            if hasattr(current, "_fsdp_wrapped_module"):
                current = current._fsdp_wrapped_module
                continue
            break
        return current

    def _get_generator_core_model(self):
        generator = self._unwrap_fsdp_like_module(self.model.generator)
        if generator is None or not hasattr(generator, "model"):
            return None
        return self._unwrap_fsdp_like_module(generator.model)

    def _get_packforcing_compressor(self):
        model = self._get_generator_core_model()
        if model is None:
            return None
        return getattr(model, "pack_compressor", None)

    def _clip_grad_norm(self, module, max_norm: float):
        if hasattr(module, "clip_grad_norm_"):
            return module.clip_grad_norm_(max_norm)
        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _packforcing_compressed_tokens_per_block(self):
        try:
            model = self._get_generator_core_model()
            if model is None or getattr(model.pack_cfg, "compress_mode", None) != "hr_spatial":
                return None
            if model.pack_compressor is None:
                return None
            latent_h = int(self.config.image_or_video_shape[-2])
            latent_w = int(self.config.image_or_video_shape[-1])
            target_h = latent_h // 8
            target_w = latent_w // 8
            return int(self.config.num_frame_per_block * target_h * target_w)
        except Exception:
            return None

    def _register_packforcing_compressor_grad_hooks(self):
        if self._packforcing_compressor_grad_hooks:
            return

        try:
            compressor = self._get_packforcing_compressor()
            if compressor is None:
                return
        except Exception:
            return

        for param in compressor.parameters():
            if not param.requires_grad:
                continue

            def _make_hook():
                def _hook(grad):
                    if grad is None:
                        return grad
                    grad_sq = grad.detach().float().pow(2).sum()
                    if (
                        self._packforcing_compressor_grad_sq_accum is None
                        or self._packforcing_compressor_grad_sq_accum.device != grad_sq.device
                    ):
                        self._packforcing_compressor_grad_sq_accum = torch.zeros(
                            (), device=grad_sq.device, dtype=torch.float32
                        )
                    self._packforcing_compressor_grad_sq_accum.add_(grad_sq)
                    return grad

                return _hook

            self._packforcing_compressor_grad_hooks.append(param.register_hook(_make_hook()))

    def _reset_packforcing_compressor_grad_stats(self):
        if self._packforcing_compressor_grad_hooks:
            self._packforcing_compressor_grad_sq_accum = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )

    def _snapshot_packforcing_compressor_params(self):
        try:
            compressor = self._get_packforcing_compressor()
            if compressor is None:
                return None
        except Exception:
            return None

        snapshot = []
        for param in compressor.parameters():
            if not param.requires_grad:
                continue
            snapshot.append(param.detach().float().clone())
        return snapshot if snapshot else None

    def _packforcing_compressor_update_norm(self, snapshot):
        if snapshot is None:
            return None
        try:
            compressor = self._get_packforcing_compressor()
            if compressor is None:
                return None
        except Exception:
            return None

        squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        snapshot_index = 0
        for param in compressor.parameters():
            if not param.requires_grad:
                continue
            if snapshot_index >= len(snapshot):
                break
            before = snapshot[snapshot_index].to(device=param.device, dtype=torch.float32)
            after = param.detach().float()
            squared_norm.add_((after - before).pow(2).sum())
            snapshot_index += 1
        if snapshot_index == 0:
            return None
        if dist.is_initialized():
            dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
        return squared_norm.sqrt()
