import types
from typing import List, Optional
from pathlib import Path
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_model_deepforcing import CausalWanModelDeepForcing
from wan.modules.causal_model_packforcing import CausalWanModelPackForcing
from wan.modules.pack_cache import PackCacheHandle


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "ckpt").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root with a ckpt directory from {start}."
    )


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_WAN_MODELS_ROOT = _REPO_ROOT / "ckpt" / "wan_models"


def wan_model_path(model_name: str, *parts: str) -> str:
    return str(_WAN_MODELS_ROOT / model_name / Path(*parts))


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(wan_model_path("Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=wan_model_path("Wan2.1-T2V-1.3B", "google", "umt5-xxl") + "/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=wan_model_path("Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                wan_model_path(model_name), local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(wan_model_path(model_name))
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        pack_cache_commit: bool = False,
        pack_cache_mode: Optional[str] = None,
        pack_cache_only: bool = False,
    ) -> torch.Tensor:
        del pack_cache_commit, pack_cache_mode, pack_cache_only
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def build_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_transformer_blocks: int,
        frame_seq_length: int,
        cache_mode: Optional[str] = None,
    ):
        del cache_mode
        kv_cache = []
        if self.model.local_attn_size != -1:
            kv_cache_size = self.model.local_attn_size * frame_seq_length
        else:
            kv_cache_size = 32760

        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        return kv_cache

    def reset_kv_cache(self, kv_cache, device: torch.device) -> None:
        for layer_cache in kv_cache:
            layer_cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            layer_cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    def commit_kv_cache(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache,
        crossattn_cache=None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
    ) -> None:
        _ = self(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
        )

    def uses_structured_kv_cache(self) -> bool:
        return False


class WanDiffusionWrapperDeepForcing(WanDiffusionWrapper):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            pc_enable=False,
            budget=16,
            recent=4,
            pc_fusion="sum",
            pc_keep_sinks=True,
            pc_topc_max_reuse=7,
            pc_mid_rope_unification=False,
            pc_bootstrap_delta=False,
    ):
        nn.Module.__init__(self)

        if is_causal:
            self.model = CausalWanModelDeepForcing.from_pretrained(
                wan_model_path(model_name),
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                PC_enable=pc_enable,
                PC_capacity=1560 * budget,
                PC_window=1560 * recent,
                PC_fusion=pc_fusion,
                PC_keep_sinks=pc_keep_sinks,
                PC_topc_max_reuse=pc_topc_max_reuse,
                PC_mid_rope_unification=pc_mid_rope_unification,
                PC_bootstrap_delta=pc_bootstrap_delta,
            )
        else:
            self.model = WanModel.from_pretrained(wan_model_path(model_name))
        self.model.eval()

        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        
        classify_mode: Optional[bool] = False, # DF
        concat_time_embeddings: Optional[bool] = False, #DF
        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep
        
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0


class WanDiffusionWrapperPackForcing(WanDiffusionWrapper):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            pack_enable=False,
            pack_sink_blocks=2,
            pack_recent_blocks=1,
            pack_mid_bank_capacity_blocks=3,
            pack_mid_select_topk_blocks=2,
            pack_mid_selection_mode="recency",
            pack_compress_mode="token_avg_pool",
            pack_compressed_tokens_per_block=1560,
            pack_compressor_detach_inputs=True,
            pack_evict_mode="fifo",
            pack_reuse_mid_selection_within_block=True,
            pack_enable_rope_adjustment=False,
    ):
        nn.Module.__init__(self)

        if is_causal:
            from_pretrained_kwargs = {}
            if pack_compress_mode == "hr_spatial":
                from_pretrained_kwargs.update(
                    {
                        "low_cpu_mem_usage": False,
                        "device_map": None,
                    }
                )
            self.model = CausalWanModelPackForcing.from_pretrained(
                wan_model_path(model_name),
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                pack_enable=pack_enable,
                pack_sink_blocks=pack_sink_blocks,
                pack_recent_blocks=pack_recent_blocks,
                pack_mid_bank_capacity_blocks=pack_mid_bank_capacity_blocks,
                pack_mid_select_topk_blocks=pack_mid_select_topk_blocks,
                pack_mid_selection_mode=pack_mid_selection_mode,
                pack_compress_mode=pack_compress_mode,
                pack_compressed_tokens_per_block=pack_compressed_tokens_per_block,
                pack_compressor_detach_inputs=pack_compressor_detach_inputs,
                pack_evict_mode=pack_evict_mode,
                pack_reuse_mid_selection_within_block=pack_reuse_mid_selection_within_block,
                pack_enable_rope_adjustment=pack_enable_rope_adjustment,
                **from_pretrained_kwargs,
            )
        else:
            self.model = WanModel.from_pretrained(wan_model_path(model_name))
        self.model.eval()

        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760
        self.post_init()

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

    def _get_pack_core_model(self):
        return self._unwrap_fsdp_like_module(self.model)

    @staticmethod
    def _resolve_pack_cache_handle(
        kv_cache,
        pack_cache_handle: PackCacheHandle | None,
    ) -> PackCacheHandle | None:
        if pack_cache_handle is not None:
            if kv_cache is not None and kv_cache is not pack_cache_handle:
                raise ValueError(
                    "Pass either kv_cache or pack_cache_handle for PackForcing, not two different handles."
                )
            return pack_cache_handle
        return kv_cache

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[PackCacheHandle] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        pack_cache_commit: bool = False,
        pack_cache_mode: Optional[str] = None,
        pack_cache_only: bool = False,
        pack_cache_handle: PackCacheHandle | None = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        pack_cache_handle = self._resolve_pack_cache_handle(kv_cache, pack_cache_handle)
        if pack_cache_handle is not None:
            if not isinstance(pack_cache_handle, PackCacheHandle):
                raise TypeError(
                    "WanDiffusionWrapperPackForcing expects pack_cache_handle to be a PackCacheHandle."
                )
            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                kv_cache=pack_cache_handle,
                pack_use_internal_kv_cache=True,
                pack_cache_session_id=pack_cache_handle.session_id,
                pack_cache_mode=pack_cache_mode or pack_cache_handle.mode,
                pack_cache_commit=pack_cache_commit,
                pack_cache_only=pack_cache_only,
            )
            if pack_cache_only:
                if model_out is not None:
                    raise RuntimeError(
                        "PackForcing cache-only forward expected the model to return None."
                    )
                return None
            flow_pred = model_out.permute(0, 2, 1, 3, 4)
        else:
            if pack_cache_only:
                raise RuntimeError(
                    "PackForcing cache-only forward requires pack_cache_handle and the internal cache path."
                )
            if clean_x is not None:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def build_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_transformer_blocks: int,
        frame_seq_length: int,
        cache_mode: Optional[str] = None,
    ):
        del batch_size, dtype, device, num_transformer_blocks, frame_seq_length
        core_model = self._get_pack_core_model()
        return core_model.create_pack_cache_handle(cache_mode=cache_mode)

    def build_pack_cache_handle(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_transformer_blocks: int,
        frame_seq_length: int,
        cache_mode: Optional[str] = None,
    ):
        return self.build_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            num_transformer_blocks=num_transformer_blocks,
            frame_seq_length=frame_seq_length,
            cache_mode=cache_mode,
        )

    def reset_kv_cache(
        self,
        kv_cache,
        device: torch.device,
        pack_cache_handle: PackCacheHandle | None = None,
    ) -> None:
        del device
        pack_cache_handle = self._resolve_pack_cache_handle(kv_cache, pack_cache_handle)
        if not isinstance(pack_cache_handle, PackCacheHandle):
            raise TypeError("PackForcing reset_kv_cache expects a PackCacheHandle.")
        core_model = self._get_pack_core_model()
        core_model.reset_pack_cache_session(pack_cache_handle)

    def reset_pack_cache_handle(
        self,
        pack_cache_handle: PackCacheHandle,
        device: torch.device,
    ) -> None:
        self.reset_kv_cache(None, device=device, pack_cache_handle=pack_cache_handle)

    def commit_kv_cache(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache=None,
        crossattn_cache=None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        pack_cache_handle: PackCacheHandle | None = None,
    ) -> None:
        _ = self(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            pack_cache_commit=True,
            pack_cache_only=True,
            pack_cache_handle=pack_cache_handle,
        )

    def commit_pack_cache_handle(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        pack_cache_handle: PackCacheHandle,
        crossattn_cache=None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
    ) -> None:
        self.commit_kv_cache(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=None,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            pack_cache_handle=pack_cache_handle,
        )

    def uses_structured_kv_cache(self) -> bool:
        return True

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
