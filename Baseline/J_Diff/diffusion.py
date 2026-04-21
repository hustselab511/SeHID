import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cct import CCT
from .layers import DualPathResBlock, TFEB, MRAB


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / max(half - 1, 1)))
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class EdBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block1 = DualPathResBlock(in_channels, in_channels)
        self.tfeb = TFEB(in_channels)
        self.block2 = DualPathResBlock(in_channels, out_channels)
        self.down = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = x + t_emb.unsqueeze(-1)
        x = self.block1(x)
        x = self.tfeb(x)
        skip = self.block2(x)
        out = self.down(skip)
        return out, skip


class DdBlock(nn.Module):
    """Paper order: up-sampling -> MRAB -> DPRB -> DPRB."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.mrab = MRAB(in_channels + skip_channels, out_channels)
        self.dprb1 = DualPathResBlock(out_channels, out_channels)
        self.dprb2 = DualPathResBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = x + t_emb.unsqueeze(-1)
        x = self.mrab(x)
        x = self.dprb1(x)
        x = self.dprb2(x)
        return x


class DiffusionRefiner(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.time_emb = TimeEmbedding(c)
        self.input_proj = nn.Conv1d(in_channels, c, 1)

        self.te_enc1 = nn.Linear(c, c)
        self.te_enc2 = nn.Linear(c, c * 2)
        self.te_enc3 = nn.Linear(c, c * 4)
        self.te_enc4 = nn.Linear(c, c * 8)
        self.te_bottleneck = nn.Linear(c, c * 16)

        self.ed1 = EdBlock(c, c * 2)
        self.ed2 = EdBlock(c * 2, c * 4)
        self.ed3 = EdBlock(c * 4, c * 8)
        self.ed4 = EdBlock(c * 8, c * 16)

        self.bottleneck = nn.Sequential(
            DualPathResBlock(c * 16, c * 16),
            DualPathResBlock(c * 16, c * 16),
        )
        self.cct = CCT(c * 16)

        self.te_dec4 = nn.Linear(c, c * 32)
        self.te_dec3 = nn.Linear(c, c * 16)
        self.te_dec2 = nn.Linear(c, c * 8)
        self.te_dec1 = nn.Linear(c, c * 4)

        self.dd4 = DdBlock(c * 16, c * 16, c * 8)
        self.dd3 = DdBlock(c * 8, c * 8, c * 4)
        self.dd2 = DdBlock(c * 4, c * 4, c * 2)
        self.dd1 = DdBlock(c * 2, c * 2, c)
        self.final_conv = nn.Conv1d(c, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, feature_c: torch.Tensor) -> torch.Tensor:
        t0 = self.time_emb(t)
        x = self.input_proj(x)
        x1, skip1 = self.ed1(x, self.te_enc1(t0))
        x2, skip2 = self.ed2(x1, self.te_enc2(t0))
        x3, skip3 = self.ed3(x2, self.te_enc3(t0))
        x4, skip4 = self.ed4(x3, self.te_enc4(t0))

        b = self.bottleneck(x4 + self.te_bottleneck(t0).unsqueeze(-1))
        b = self.cct(b, feature_c)

        d4 = self.dd4(b, skip4, self.te_dec4(t0))
        d3 = self.dd3(d4, skip3, self.te_dec3(t0))
        d2 = self.dd2(d3, skip2, self.te_dec2(t0))
        d1 = self.dd1(d2, skip1, self.te_dec1(t0))
        return self.final_conv(d1)
