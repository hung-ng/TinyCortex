"""
TinyCortex — Causal Dilated TCN for Neural Amp Modeling

Architecture: Input → 1×1 Conv → [TCN Block × N] → 1×1 Conv → Output
Each TCN Block: Causal Dilated Conv1d → PReLU → 1×1 Conv → Residual Add

TCN chosen over LSTM because it decomposes into simple 1D convolutions,
which are straightforward to implement as Metal compute kernels.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad so output[t] depends only on input[<=t]
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Dilated causal conv → PReLU → 1×1 conv → residual add."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.PReLU(num_parameters=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyCortexTCN(nn.Module):
    """Causal dilated TCN for guitar amp modeling.

    Args:
        n_channels: Hidden channel width (default 16).
        n_blocks: Number of TCN blocks with exponentially increasing dilation.
        kernel_size: Kernel size for dilated convolutions.
    """

    def __init__(self, n_channels: int = 16, n_blocks: int = 10, kernel_size: int = 3):
        super().__init__()
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size

        self.input_conv = nn.Conv1d(1, n_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TCNBlock(n_channels, kernel_size, dilation=2**i)
                for i in range(n_blocks)
            ]
        )
        self.output_conv = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 1, T) → (B, 1, T)"""
        x = self.input_conv(x)
        x = self.blocks(x)
        return self.output_conv(x)

    def receptive_field(self) -> int:
        """Total receptive field in samples."""
        # Each block adds (kernel_size - 1) * dilation to the receptive field
        return 1 + sum(
            (self.kernel_size - 1) * (2**i) for i in range(self.n_blocks)
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
