import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathResBlock(nn.Module):
    """
    Paper-faithful Dual-path Res-Block.
    Standard branch: 1x1 -> k7 -> k5
    Dilated branch: 1x1 -> dilated k5 -> dilated k7
    The two branches are concatenated and fused by 1x1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even for DualPathResBlock.")
        mid_channels = out_channels // 2

        self.std_proj = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.std_body = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )

        self.dil_proj = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.dil_body = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=7,
                padding=3 * dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = self.std_proj(x)
        std = std + self.std_body(std)

        dil = self.dil_proj(x)
        dil = dil + self.dil_body(dil)

        return self.fuse(torch.cat([std, dil], dim=1))


class GCAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(x, 1) + F.adaptive_max_pool1d(x, 1)
        attn = self.mlp(pooled)
        return x * attn


class LCAM(nn.Module):
    """Implements Eq. (17): GAP-based channel attention multiplied with local features."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Conv1d(channels, channels, 5, padding=2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x) * self.attn(x)


class TSAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.reduce = nn.Conv1d(channels, hidden, 1, bias=False)
        self.compress = nn.Conv1d(hidden, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.compress(F.gelu(self.reduce(x)))
        attn = torch.sigmoid(attn)
        return x * attn


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 1)
        attn, _ = self.mha(y, y, y)
        y = self.norm(y + attn)
        return y.permute(0, 2, 1)


class TFEB(nn.Module):
    """
    Time-Frequency Enhancement Block.
    Time branch: GCAM -> TSAM
    Frequency branch: FFT -> real/imag 1x1 convs -> IFFT -> 1x1 conv -> self-attn
    Final output is concatenation of two branches to match the paper description.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gcam = GCAM(channels)
        self.tsam = TSAM(channels)

        self.real_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=False),
        )
        self.imag_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=False),
        )
        self.post_ifft = nn.Conv1d(channels, channels, 1, bias=False)
        self.sa = SelfAttention(channels)
        self.fuse = nn.Conv1d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_time = self.tsam(self.gcam(x))

        fft_x = torch.fft.rfft(x, dim=-1, norm="ortho")
        real = self.real_conv(fft_x.real)
        imag = self.imag_conv(fft_x.imag)
        x_freq = torch.fft.irfft(torch.complex(real, imag), n=x.shape[-1], dim=-1, norm="ortho")
        x_freq = self.post_ifft(x_freq)
        x_freq = self.sa(x_freq + x)

        return self.fuse(torch.cat([x_time, x_freq], dim=1))


class MRAB(nn.Module):
    """
    Multiresolution Attention Block using depth-wise separable convolutions
    and the three attention modules GCAM / LCAM / TSAM.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        c = out_channels

        self.stem = nn.Sequential(
            nn.BatchNorm1d(c),
            nn.Conv1d(c, c, 1, bias=False),
            nn.Conv1d(c, c, 5, padding=2, bias=False),
        )

        self.dw3 = nn.Conv1d(c, c, 3, padding=1, groups=c, bias=False)
        self.dw5 = nn.Conv1d(c, c, 5, padding=2, groups=c, bias=False)
        self.dw7 = nn.Conv1d(c, c, 7, padding=3, groups=c, bias=False)
        self.dw9 = nn.Conv1d(c, c, 9, padding=4, groups=c, bias=False)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(4 * c, 5 * c, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(5 * c, c, 1, bias=False),
        )

        self.bn = nn.BatchNorm1d(c)
        self.tsam = TSAM(c)
        self.lcam = LCAM(c)
        self.gcam = GCAM(c)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(3 * c, 4 * c, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(4 * c, c, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        identity = x

        stem = self.stem(x)
        branches = [self.dw3(stem), self.dw5(stem), self.dw7(stem), self.dw9(stem)]
        x_mid = self.mlp1(torch.cat(branches, dim=1)) + identity

        x_bn = self.bn(x_mid)
        attn = torch.cat([self.gcam(x_bn), self.lcam(x_bn), self.tsam(x_bn)], dim=1)
        return self.mlp2(attn) + x_mid
