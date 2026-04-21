import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding used in the paper.
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_len {self.pe.size(1)}."
            )
        return x + self.pe[:, :seq_len, :]


class ECG_Trans(nn.Module):
    """
    Reproduction of the Transformer module in:
    "Heartbeat Detection from Ballistocardiogram using Transformer Network"

    Paper-side core architecture:
      1) input linear layer
      2) sinusoidal positional encoding
      3) 4-layer Transformer encoder, 8 heads, FFN dim 2048
      4) output fully connected layer

    This implementation keeps the user's I/O convention unchanged:
      input : (batch, 1, seq_len)
      output: (batch, 1, seq_len)

    Note:
    - The paper uses 5-second windows at 100 Hz, i.e. seq_len = 500.
    - This implementation supports arbitrary seq_len up to max_len so it can
      preserve the user's existing pipeline (for example seq_len = 625 at 125 Hz).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, bcg_signal: torch.Tensor) -> torch.Tensor:
        if bcg_signal.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, 1, seq_len), got {tuple(bcg_signal.shape)}"
            )
        if bcg_signal.size(1) != 1:
            raise ValueError(
                f"Expected single-channel BCG input, got {bcg_signal.size(1)} channels"
            )

        x = bcg_signal.permute(0, 2, 1)          # (B, T, 1)
        x = self.input_proj(x)                   # (B, T, d_model)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)                  # (B, T, d_model)
        x = self.transformer_encoder(x)          # (B, T, d_model)
        pred_ecg = self.output_proj(x)           # (B, T, 1)
        pred_ecg = pred_ecg.permute(0, 2, 1)     # (B, 1, T)
        return pred_ecg


if __name__ == "__main__":
    x = torch.rand(1, 1, 625)
    model = ECG_Trans()
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
