import torch
import torch.nn as nn
from .layers import DualPathResBlock, MRAB


class ConditionalDetector(nn.Module):
    """
    Conditional detector from Fig. 2(e):
    Ec blocks: DualPathResBlock + downsampling
    Dc blocks: upsampling + MRAB
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.enc1 = DualPathResBlock(in_channels, c)
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = DualPathResBlock(c, c * 2)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = DualPathResBlock(c * 2, c * 4)
        self.down3 = nn.MaxPool1d(2)
        self.enc4 = DualPathResBlock(c * 4, c * 8)
        self.down4 = nn.MaxPool1d(2)
        self.bottleneck = DualPathResBlock(c * 8, c * 16)

        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec4 = MRAB(c * 16 + c * 8, c * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec3 = MRAB(c * 8 + c * 4, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec2 = MRAB(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec1 = MRAB(c * 2 + c, c)
        self.final_conv = nn.Conv1d(c, 1, 1)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        f_c = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(f_c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        mask = self.final_conv(d1)
        return mask, f_c
