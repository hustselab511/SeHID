
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding used by the paper's Attention module.
    Input/Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}.")
        return x + self.pe[:, :seq_len, :]


class AttentionModule(nn.Module):
    """
    Paper-faithful attention branch:
    - embedding: 1 -> d_model
    - positional encoding
    - 2-layer Transformer encoder
    - output wide R-peak-like 1D signal
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        num_heads: int = 32,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_layer(x)          # (batch, seq_len, 1)
        return x.permute(0, 2, 1)        # (batch, 1, seq_len)


class PhaseShiftModule(nn.Module):
    """
    Training-only phase shift module from the paper.
    It estimates a scalar shift amount theta from:
        [Attention output, BCG, ECG] -> theta
    During inference, this module is bypassed.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 10, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(96),  # keeps the FC interface stable
        )
        self.fc1 = nn.Linear(10 * 96, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    @staticmethod
    def _shift_1d(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Integer shift using torch.roll.
        x: (batch, channels, seq_len)
        shift: (batch, 1) or (batch,)
        """
        batch = x.size(0)
        out = []
        shift = shift.view(batch).round().long()
        for i in range(batch):
            out.append(torch.roll(x[i], shifts=int(shift[i].item()), dims=-1))
        return torch.stack(out, dim=0)

    def forward(
        self,
        attn_signal: torch.Tensor,
        bcg_signal: torch.Tensor,
        ecg_signal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        montage = torch.cat([attn_signal, bcg_signal, ecg_signal], dim=1)  # (B,3,T)
        feat = self.conv_block(montage)
        feat = feat.flatten(start_dim=1)
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        theta = self.fc2(feat)  # (B,1)

        shifted_attn = self._shift_1d(attn_signal, theta)
        shifted_bcg = self._shift_1d(bcg_signal, theta)
        return shifted_attn, shifted_bcg, theta


class InputResidualBlock(nn.Module):
    """
    Paper input layer:
    main path: Conv(k=17) -> BN -> ReLU -> Conv(k=17)
    skip path: Conv(k=1) -> BN
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + self.skip(x), inplace=False)


class ResidualBlock(nn.Module):
    """
    Paper residual block:
    BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    skip path: 1x1 Conv -> BN
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=17, padding=8, bias=False),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class UpBlock(nn.Module):
    """
    Paper up-sampling style:
    linear interpolation -> 1x1 conv to halve channels -> concatenate skip -> residual block
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    """
    Paper-faithful ResUNet branch:
    input: [attention, bcg] -> 2 channels
    contracting path with 3 max-pools
    Bi-LSTM at the bottleneck
    expanding path with interpolation + 1x1 conv
    output: narrow R-peak-like signal
    """

    def __init__(self, input_channels: int = 2, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.input_block = InputResidualBlock(input_channels, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)

        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)
        self.final_dropout = nn.Dropout(dropout_p)
        self.final_relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.input_block(x)   # (B,32,T)
        p1 = self.pool1(enc1)        # (B,32,T/2)

        enc2 = self.enc2(p1)         # (B,64,T/2)
        p2 = self.pool2(enc2)        # (B,64,T/4)

        enc3 = self.enc3(p2)         # (B,128,T/4)
        p3 = self.pool3(enc3)        # (B,128,T/8)

        # Bi-LSTM at bottleneck
        lstm_in = p3.permute(0, 2, 1)          # (B,T/8,128)
        lstm_out, _ = self.bilstm(lstm_in)     # (B,T/8,256)
        x = lstm_out.permute(0, 2, 1)          # (B,256,T/8)

        x = self.up3(x, enc3)                  # (B,128,T/4)
        x = self.up2(x, enc2)                  # (B,64,T/2)
        x = self.up1(x, enc1)                  # (B,32,T)

        x = self.final_conv(x)
        x = self.final_dropout(x)
        x = self.final_relu(x)

        # The paper explicitly describes an extra test-time scaling after Dropout.
        # PyTorch already handles training/eval scaling internally, so we keep the
        # standard behavior by default to preserve stable inference.
        return x


class R_RecNet(nn.Module):
    """
    Updated paper-faithful model while preserving the public output interface:
        input  : (batch, 1, seq_len)
        output : pred, attn_pred
                 pred      -> reconstructed narrow R-wave signal, shape (batch,1,seq_len)
                 attn_pred -> wide R-wave-like signal,        shape (batch,1,seq_len)

    Notes:
    - Signal preprocessing is assumed to be performed upstream.
    - The paper's Phase Shift module is training-only. It is implemented here,
      but the default forward() remains inference-style to preserve the interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention = AttentionModule()
        self.phase_shift = PhaseShiftModule()
        self.resunet = ResUNet(input_channels=2)

    def forward(self, bcg_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if bcg_signal.ndim != 3 or bcg_signal.size(1) != 1:
            raise ValueError(
                f"Expected bcg_signal with shape (batch, 1, seq_len), got {tuple(bcg_signal.shape)}"
            )

        bcg_for_attn = bcg_signal.permute(0, 2, 1)   # (B,T,1)
        attn_pred = self.attention(bcg_for_attn)     # (B,1,T)

        combined = torch.cat([attn_pred, bcg_signal], dim=1)  # (B,2,T)
        pred = self.resunet(combined)                          # (B,1,T)
        return pred, attn_pred

    def forward_train(
        self,
        bcg_signal: torch.Tensor,
        ecg_signal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Paper-style training path with phase shift module enabled.
        Returns:
            pred, attn_pred, shifted_attn, shifted_bcg, theta
        """
        if bcg_signal.shape != ecg_signal.shape:
            raise ValueError("bcg_signal and ecg_signal must have the same shape.")

        bcg_for_attn = bcg_signal.permute(0, 2, 1)
        attn_pred = self.attention(bcg_for_attn)

        shifted_attn, shifted_bcg, theta = self.phase_shift(attn_pred, bcg_signal, ecg_signal)
        combined = torch.cat([shifted_attn, shifted_bcg], dim=1)
        pred = self.resunet(combined)
        return pred, attn_pred, shifted_attn, shifted_bcg, theta

    @staticmethod
    def gaussian_r_target(
        r_indices: torch.Tensor,
        seq_len: int,
        sigma: float,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Build Gaussian-shaped R-wave targets described in the paper.
        r_indices: (batch, num_peaks) integer indices, padded entries should be <0
        return   : (batch, 1, seq_len)
        """
        if device is None:
            device = r_indices.device

        t = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, 1, seq_len)
        peaks = r_indices.float().unsqueeze(-1)  # (B,N,1)
        valid = (r_indices >= 0).float().unsqueeze(-1)
        gauss = torch.exp(-((t - peaks) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
        gauss = gauss * valid
        return gauss.sum(dim=1, keepdim=True)

    @staticmethod
    def compute_losses(
        pred: torch.Tensor,
        attn_pred: torch.Tensor,
        shifted_attn: Optional[torch.Tensor],
        wide_target: torch.Tensor,
        narrow_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Paper-consistent loss decomposition:
            L_attn    : wide target supervision on attention output
            L_psn     : wide target supervision after phase shift
            L_resunet : narrow target supervision on final output
            L_total   : sum of the above
        """
        l_attn = F.mse_loss(attn_pred, wide_target)
        if shifted_attn is None:
            l_psn = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            l_psn = F.mse_loss(shifted_attn, wide_target)
        l_resunet = F.mse_loss(pred, narrow_target)
        l_total = l_attn + l_psn + l_resunet
        return l_total, l_attn, l_psn, l_resunet


if __name__ == "__main__":
    dummy_bcg = torch.rand(1, 1, 625)

    model = R_RecNet()
    pred, attn_pred = model(dummy_bcg)

    print("Input shape:", dummy_bcg.shape)
    print("Attention output shape:", attn_pred.shape)
    print("Prediction shape:", pred.shape)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
