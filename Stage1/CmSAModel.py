import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

__all__ = ['CmSAModel']


# ==========================================
# Helper Functions
# ==========================================

def build_three_layer_conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """
    Constructs a standard three-layer convolution block:
    Conv-BN-ReLU -> Conv-BN-ReLU -> Conv-BN.
    Used for Q/K/V projections or feature extraction.
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm1d(out_ch)
    )


class PositionalEncoding(nn.Module):
    """
    Lightweight Sinusoidal Positional Encoding.
    Adds temporal information to the feature embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 1250):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Time, Dim]
        # Slice to match the sequence length of x
        return x + self.pe[:x.size(1)]


class InitialConvBlock(nn.Module):
    """
    Initial feature extraction block (Upscaling channel dimensions).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


# ==========================================
# Core Modules
# ==========================================

class SelfAttentionBlock(nn.Module):
    """
    Single Self-Attention Block using Convolutional Projections.
    """

    def __init__(self, input_dim: int, d_model: int, num_heads: int,
                 dropout: float, max_len: int, ffn_dim: int):
        super().__init__()
        self.model_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Feature Projection
        self.conv_encoder = InitialConvBlock(in_channels=input_dim, out_channels=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Q/K/V Projections (using Conv1d to preserve temporal features)
        self.Q_conv = build_three_layer_conv(d_model, d_model)
        self.K_conv = build_three_layer_conv(d_model, d_model)
        self.V_conv = build_three_layer_conv(d_model, d_model)

        # Attention & FFN layers
        self.attn_dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 1. Conv Encoding: [B, T, 1] -> [B, D, T] for Conv layers
        x_conv = x.permute(0, 2, 1)
        x_enc = self.conv_encoder(x_conv)  # [B, D, T]

        # 2. Positional Encoding: [B, D, T] -> [B, T, D]
        x_emb = x_enc.permute(0, 2, 1)
        x_emb = self.pos_encoder(x_emb)

        # 3. QKV Projections (Input requires [B, D, T])
        x_for_proj = x_emb.permute(0, 2, 1)
        Q = self.Q_conv(x_for_proj).permute(0, 2, 1)  # [B, T, D]
        K = self.K_conv(x_for_proj).permute(0, 2, 1)
        V = self.V_conv(x_for_proj).permute(0, 2, 1)

        # 4. Multi-Head Attention Calculation
        # Reshape to [B, Heads, T, Head_Dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.attention_weights = attn_weights  # Save for visualization

        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Merge Heads: [B, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        # 5. Add & Norm (Residual Connection)
        x = self.norm1(x_emb + self.dropout1(attn_output))

        # 6. FFN & Norm (Residual Connection)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class CmSAModel(nn.Module):
    """
    Stage 1: CmSAModel (Pure Attention Architecture).
    Used for initial feature extraction and estimation.
    """

    def __init__(self, input_dim: int = 1, d_model: int = 32, num_heads: int = 4,
                 dropout: float = 0.3, max_len: int = 625, ffn_dim: int = 128):
        super().__init__()

        # Stack 2 Layers of Attention
        self.layer1 = SelfAttentionBlock(input_dim, d_model, num_heads, dropout, max_len, ffn_dim)
        self.layer2 = SelfAttentionBlock(d_model, d_model, num_heads, dropout, max_len, ffn_dim)

        self.output_head = nn.Linear(d_model, 1)

    def forward(self, bcg_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bcg_signal: Input signal tensor (B, 1, T)

        Returns:
            features: [B, T, d_model] (Extracted features for Stage 2)
            prediction: [B, 1, T] (Stage 1 initial prediction)
        """
        # Input: [B, 1, T] -> [B, T, 1] for Attention
        x = bcg_signal.permute(0, 2, 1)

        # Pass through layers
        feat1 = self.layer1(x)
        features = self.layer2(feat1)  # This serves as "feature_32d" for Stage 2

        # Output prediction
        pred = self.output_head(features)  # [B, T, 1]
        prediction = pred.permute(0, 2, 1)  # [B, 1, T]

        return features, prediction


if __name__ == "__main__":
    # Test Block
    model = CmSAModel(max_len=625)
    dummy = torch.randn(2, 1, 625)
    feats, pred = model(dummy)
    print(f"CmSAModel - Feature Shape: {feats.shape}, Prediction Shape: {pred.shape}")