import torch
import torch.nn as nn
import torch.nn.functional as F


class CCT_FFN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(self.act(self.conv1(x)))
        return x + residual


class CCT(nn.Module):
    """Conditional Cross Transformer implementing Eqs. (19)-(22)."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels ** -0.5
        self.proj_d = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj_c = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj_c1 = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.ffn1 = CCT_FFN(channels)
        self.ffn2 = CCT_FFN(channels)

    def forward(self, F_d: torch.Tensor, F_c: torch.Tensor) -> torch.Tensor:
        if F_c.shape[-1] != F_d.shape[-1]:
            F_c = F.interpolate(F_c, size=F_d.shape[-1], mode="linear", align_corners=False)

        q_d, k_d, v_d = self.proj_d(F_d).chunk(3, dim=1)
        q_c, k_c, v_c = self.proj_c(F_c).chunk(3, dim=1)

        attn1 = torch.softmax((q_d.transpose(1, 2) @ k_c.transpose(1, 2).transpose(-2, -1)) * self.scale, dim=-1)
        x1 = (attn1 @ v_c.transpose(1, 2)).transpose(1, 2)
        F_c1 = self.ffn1(x1)

        q_c1, _, _ = self.proj_c1(F_c1).chunk(3, dim=1)
        attn2 = torch.softmax((q_c1.transpose(1, 2) @ k_d.transpose(1, 2).transpose(-2, -1)) * self.scale, dim=-1)
        x2 = (attn2 @ v_d.transpose(1, 2)).transpose(1, 2) + F_c1
        return self.ffn2(x2)
