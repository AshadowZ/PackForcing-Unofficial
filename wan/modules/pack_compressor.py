from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PackHRSpatialCompressorConfig:
    latent_channels: int = 16
    hidden_dim: int = 1536
    stage_channels: tuple[int, int, int] = (64, 256, 1024)
    detach_inputs: bool = True


class PackHRSpatialCompressor(nn.Module):
    """HR-only PackForcing compressor with pure spatial downsampling.

    The module keeps the temporal axis intact and applies three spatial
    downsampling stages with stride `(1, 2, 2)`. The flattened output preserves
    the native `Time -> Height -> Width` token order used by Wan patch
    embeddings so RoPE indexing stays aligned with the existing cache logic.
    """

    compressor_kind = "hr_spatial"

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 1536,
        stage_channels: tuple[int, int, int] = (64, 256, 1024),
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError(
                "PackHRSpatialCompressor expects exactly three stage channels, "
                f"got {stage_channels}."
            )

        self.cfg = PackHRSpatialCompressorConfig(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            stage_channels=stage_channels,
            detach_inputs=detach_inputs,
        )

        c1, c2, c3 = stage_channels
        self.net = nn.Sequential(
            nn.Conv3d(
                latent_channels,
                c1,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                c1,
                c2,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                c2,
                c3,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(c3, hidden_dim, kernel_size=1),
        )

    def compress_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 5:
            raise ValueError(
                "PackHRSpatialCompressor expects latent shaped [B, C, T, H, W], "
                f"got {tuple(latent.shape)}."
            )

        if self.cfg.detach_inputs:
            latent = latent.detach()

        batch_size, channels, num_frames, height, width = latent.shape
        if channels != self.cfg.latent_channels:
            raise ValueError(
                f"Expected latent_channels={self.cfg.latent_channels}, got {channels}."
            )

        x = self.net(latent)
        target_t = num_frames
        target_h = max(1, height // 8)
        target_w = max(1, width // 8)
        x = F.adaptive_avg_pool3d(x, (target_t, target_h, target_w))

        # Keep the native patch-token traversal order: Time -> Height -> Width.
        hidden = x.flatten(2).transpose(1, 2).contiguous()
        grid_sizes = torch.tensor(
            [[target_t, target_h, target_w]] * batch_size,
            device=latent.device,
            dtype=torch.long,
        )
        return hidden, grid_sizes
