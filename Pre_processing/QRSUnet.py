import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ['QRSUNet']


class ResidualBlock(nn.Module):
    """
    1D Residual Block: (Conv-BN-ReLU-Conv-BN) + Residual Connection.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class AttentionBlock(nn.Module):
    """
    Attention Gate for U-Net.
    Filters the features from the skip connection (x) using the gating signal (g)
    from the coarser scale.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in the gating signal (g).
            F_l: Number of channels in the skip connection (x).
            F_int: Number of intermediate channels.
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: Gating signal (from lower decoder layer)
        # x: Skip connection (from encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Resize g1 to match x1 if there is a size mismatch (common with MaxPool ceil_mode)
        if g1.size(2) != x1.size(2):
            g1 = F.interpolate(g1, size=x1.size(2), mode='linear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class QRSUNet(nn.Module):
    """
    1D U-Net with Residual Blocks and Attention Gates for QRS Segmentation.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(QRSUNet, self).__init__()

        # --- Encoder Path ---
        self.encoder1 = ResidualBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)

        self.encoder2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2, ceil_mode=True)

        self.encoder3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2, ceil_mode=True)

        # --- Bottleneck ---
        self.middle = ResidualBlock(128, 256)

        # --- Decoder Path ---
        # Level 3
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(256, 128)  # Input: 128 (skip) + 128 (up) = 256
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)

        # Level 2
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(128, 64)  # Input: 64 (skip) + 64 (up) = 128
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # Level 1
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(64, 32)  # Input: 32 (skip) + 32 (up) = 64
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)

        # --- Output Head ---
        self.final_conv = nn.Conv1d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # 2. Bottleneck
        middle = self.middle(self.pool3(enc3))

        # 3. Decoder
        # Level 3 Decoding
        dec3 = self.upconv3(middle)
        att3 = self.att3(g=dec3, x=enc3)  # Attention gating

        # Align dimensions for concatenation
        if dec3.size(2) != att3.size(2):
            dec3 = F.interpolate(dec3, size=att3.size(2), mode='linear', align_corners=True)

        merge3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(merge3)

        # Level 2 Decoding
        dec2 = self.upconv2(dec3)
        att2 = self.att2(g=dec2, x=enc2)

        if dec2.size(2) != att2.size(2):
            dec2 = F.interpolate(dec2, size=att2.size(2), mode='linear', align_corners=True)

        merge2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(merge2)

        # Level 1 Decoding
        dec1 = self.upconv1(dec2)
        att1 = self.att1(g=dec1, x=enc1)

        if dec1.size(2) != att1.size(2):
            dec1 = F.interpolate(dec1, size=att1.size(2), mode='linear', align_corners=True)

        merge1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(merge1)

        # Output
        out = self.final_conv(dec1)

        # Ensure output size exactly matches input size
        if out.size(2) != x.size(2):
            out = F.interpolate(out, size=x.size(2), mode='linear', align_corners=True)

        return out


if __name__ == "__main__":
    # Test Block
    model = QRSUNet(in_channels=1, num_classes=2)
    # Test with standard length
    dummy_input = torch.randn(1, 1, 625)
    output = model(dummy_input)
    print(f"QRSUNet - Input: {dummy_input.shape}, Output: {output.shape}")