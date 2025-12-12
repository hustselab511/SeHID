import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['FGWRModel']


# ==========================================
# Helper Functions
# ==========================================

def build_three_layer_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Reuses the three-layer convolution logic from Stage 1 for CrossAttention projections.
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm1d(out_ch)
    )


class ResidualConvBlock(nn.Module):
    """
    Standard Residual Convolutional Block with BatchNorm and ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.block(x)
        return self.relu(out + res)


class DownsampleBlock(nn.Module):
    """
    Convolutional Downsampling Block (Stride=2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.block(x)


# ==========================================
# Core Modules
# ==========================================

class CrossAttentionBlock(nn.Module):
    """
    Dual-Path Cross Attention Block.
    Query: Main BCG Path Features.
    Key/Value: Auxiliary Feature Path (from Stage 1).
    """

    def __init__(self, embed_dim: int, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Projections
        self.Q_conv = build_three_layer_conv(embed_dim, embed_dim)
        self.K_conv = build_three_layer_conv(feature_dim, embed_dim)
        self.V_conv = build_three_layer_conv(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        # x: [B, C_main, L]
        # feature: [B, C_feat, L]
        batch_size = x.size(0)

        # Projections: [B, C, L] -> [B, L, E]
        Q = self.Q_conv(x).permute(0, 2, 1)
        K = self.K_conv(feature).permute(0, 2, 1)
        V = self.V_conv(feature).permute(0, 2, 1)

        # Multi-head Split: [B, Heads, L, Head_Dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Calculation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))

        # Aggregation
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Residual + Norm (Add & Norm)
        # Note: Residual connection is based on the Query (Main Path)
        orig_Q = Q.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.norm(orig_Q + self.dropout(attn_out))

        return out.permute(0, 2, 1)  # Return [B, C, L]


class FeaturePathEncoder(nn.Module):
    """
    Dedicated encoder for the auxiliary feature path.
    """

    def __init__(self, input_channels: int = 32):
        super().__init__()
        self.layer1 = ResidualConvBlock(input_channels, 32)

        self.layer2_up = ResidualConvBlock(32, 64)
        self.layer2_down = DownsampleBlock(64)

        self.layer3_up = ResidualConvBlock(64, 128)
        self.layer3_down = DownsampleBlock(128)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2_down(self.layer2_up(f1))
        f3 = self.layer3_down(self.layer3_up(f2))
        return f1, f2, f3


class DualPathResUNet(nn.Module):
    """
    Stage 2 Core: Dual-Path Residual U-Net with Cross Attention & Bi-LSTM.
    """

    def __init__(self, feature_dim: int = 32, num_heads: int = 4):
        super().__init__()

        # 1. Feature Path Encoder
        self.feat_enc = FeaturePathEncoder(feature_dim)

        # 2. Main BCG Path (Encoder)
        self.bcg_enc1 = ResidualConvBlock(1, 32)
        self.ca1 = CrossAttentionBlock(32, 32, num_heads)
        self.down1 = DownsampleBlock(32)

        self.bcg_enc2 = ResidualConvBlock(32, 64)
        self.ca2 = CrossAttentionBlock(64, 64, num_heads)
        self.down2 = DownsampleBlock(64)

        self.bcg_enc3 = ResidualConvBlock(64, 128)
        self.ca3 = CrossAttentionBlock(128, 128, num_heads)
        self.down3 = DownsampleBlock(128)

        # 3. Bottleneck (Bi-LSTM)
        self.lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True)

        # 4. Decoder
        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ResidualConvBlock(128 + 128, 128), nn.Dropout(0.2))

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ResidualConvBlock(64 + 64, 64), nn.Dropout(0.2))

        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ResidualConvBlock(32 + 32, 32), nn.Dropout(0.2))

        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, bcg, feature):
        L = bcg.size(2)

        # --- Encoder: Feature Path ---
        f1, f2, f3 = self.feat_enc(feature)

        # --- Encoder: Main Path ---
        # Level 1
        x1 = self.ca1(self.bcg_enc1(bcg), f1)
        x1_d = self.down1(x1)

        # Level 2
        x2 = self.ca2(self.bcg_enc2(x1_d), f2)
        x2_d = self.down2(x2)

        # Level 3
        x3 = self.ca3(self.bcg_enc3(x2_d), f3)
        x3_d = self.down3(x3)

        # --- Bottleneck ---
        lstm_in = x3_d.permute(0, 2, 1)  # [B, L, C]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1)  # [B, C, L] (C=256)

        # --- Decoder ---
        # Level 3
        d3 = self.up3(lstm_out)
        # Skip Connection (Crop to min length to handle odd input sizes)
        len3 = min(d3.size(2), x3.size(2))
        d3 = torch.cat([d3[:, :, :len3], x3[:, :, :len3]], dim=1)
        d3 = self.dec3(d3)

        # Level 2
        d2 = self.up2(d3)
        len2 = min(d2.size(2), x2.size(2))
        d2 = torch.cat([d2[:, :, :len2], x2[:, :, :len2]], dim=1)
        d2 = self.dec2(d2)

        # Level 1
        d1 = self.up1(d2)
        len1 = min(d1.size(2), x1.size(2))
        d1 = torch.cat([d1[:, :, :len1], x1[:, :, :len1]], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.final(d1)
        # Interpolate to ensure exact output length match with input
        return F.interpolate(out, size=L, mode='linear', align_corners=True)


class FGWRModel(nn.Module):
    """
    Stage 2: FGWRModel (Wrapper).
    """

    def __init__(self, feature_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.model = DualPathResUNet(feature_dim, num_heads)

    def forward(self, feature_32d: torch.Tensor, bcg_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_32d: Features from Stage 1 [B, T, 32]
            bcg_signal: Raw BCG signal [B, 1, T]
        """
        # Transpose feature for CNN format: [B, T, 32] -> [B, 32, T]
        feat = feature_32d.permute(0, 2, 1)
        return self.model(bcg_signal, feat)


if __name__ == "__main__":
    # Test Block
    model = FGWRModel()
    dummy_feat = torch.randn(2, 625, 32)
    dummy_bcg = torch.randn(2, 1, 625)
    out = model(dummy_feat, dummy_bcg)
    print(f"FGWRModel - Output Shape: {out.shape}")