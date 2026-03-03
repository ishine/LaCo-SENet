"""
Reshape-Free Layers for Batch=1 Optimized Inference.

This module provides Conv2d-based alternatives to Conv1d layers that eliminate
the need for reshape operations in TS_BLOCK processing. When batch_size=1,
this approach provides up to 193x speedup by avoiding memory copies.

Key insight:
    Conv1d on reshaped [B*F, C, T] ≡ Conv2d with kernel (K, 1) on [B, C, T, F]
    Conv1d on reshaped [B*T, C, F] ≡ Conv2d with kernel (1, K) on [B, C, T, F]

Usage:
    >>> from src.models.streaming.layers.reshape_free import ReshapeFreeTSBlock
    >>> ts_block = ReshapeFreeTSBlock(dense_channel=64)
    >>> x = torch.randn(1, 64, 40, 100)  # [B, C, T, F]
    >>> out = ts_block(x)  # No reshape operations!

Reference:
    See android/docs/BATCH1_OPTIMIZATION_STRATEGY.md for detailed analysis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChannelLayerNorm2d(nn.Module):
    """
    Channel-wise layer normalization for 4D tensors [B, C, T, F].

    Normalizes over the channel dimension (dim=1), matching the behavior of
    LayerNorm1d on reshaped 3D tensors:
        LayerNorm1d on [B*F, C, T]  ≡  ChannelLayerNorm2d on [B, C, T, F]

    This is the correct 4D equivalent for converting original Backbone's
    LayerNorm1d to reshape-free 4D processing.

    Args:
        channels: Number of channels
        eps: Small constant for numerical stability
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # Learnable parameters [1, C, 1, 1] for broadcasting
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Normalized tensor [B, C, T, F]
        """
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, T, F]
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)  # [B, 1, T, F]
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


class SimpleGate2d(nn.Module):
    """SimpleGate for 4D tensors [B, C, T, F]."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ReshapeFreeCAB(nn.Module):
    """
    Reshape-Free Channel Attention Block.

    Replaces the original CAB that requires reshaping by using Conv2d
    with axis-specific kernels.

    Original CAB flow (with reshape):
        [B,C,T,F] → reshape → [B*F,C,T] → Conv1d → reshape → [B,C,T,F]

    Reshape-Free flow:
        [B,C,T,F] → Conv2d(kernel=(K,1)) → [B,C,T,F]  (no reshape!)

    Args:
        channels: Number of input/output channels
        kernel_size: Kernel size for depthwise conv
        axis: Processing axis ('time' or 'freq')
        causal: If True and axis='time', use causal (left-only) padding
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        axis: str = "time",
        causal: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_size = kernel_size
        self.causal = causal and (axis == "time")  # causal only applies to time axis
        dw_channel = channels * 2

        self.norm = ChannelLayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, dw_channel, kernel_size=1)

        # Axis-specific depthwise conv
        if axis == "time":
            # Process along time axis: kernel (K, 1)
            # For causal: no padding in conv, manual left padding in forward
            padding = 0 if self.causal else kernel_size // 2
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                groups=dw_channel,
            )
            # Causal padding amount: (kernel_size - 1) on left
            self._causal_pad = kernel_size - 1
        else:
            # Process along freq axis: kernel (1, K)
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dw_channel,
            )

        self.sg = SimpleGate2d()

        # SCA: squeeze (mean) + channel attention conv
        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Output tensor [B, C, T, F]
        """
        skip = x

        x = self.norm(x)
        x = self.pwconv1(x)

        # Apply causal padding if needed
        if self.causal:
            # Pad on left side of time axis: F.pad uses (left, right, top, bottom) for last 2 dims
            # For [B, C, T, F], time is dim 2, so we pad (0, 0, _causal_pad, 0)
            x = F.pad(x, (0, 0, self._causal_pad, 0))

        x = self.dwconv(x)
        x = self.sg(x)

        # SCA: global average along processing axis
        if self.axis == "time":
            attn = x.mean(dim=2, keepdim=True)  # [B, C, 1, F]
        else:
            attn = x.mean(dim=3, keepdim=True)  # [B, C, T, 1]

        attn = self.sca_conv(attn)
        x = x * attn

        x = self.pwconv2(x)

        return skip + x * self.beta


class CausalConv2dTime(nn.Module):
    """Causal Conv2d for time axis with left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pad left side of time axis only
        x = F.pad(x, (0, 0, self.padding, 0))
        return self.conv(x)


class ReshapeFreeGPKFFN(nn.Module):
    """
    Reshape-Free Group Prime Kernel FFN.

    Replaces the original GPKFFN that requires reshaping by using Conv2d
    with axis-specific kernels for each prime kernel size.

    Original GPKFFN flow:
        [B,C,T,F] → reshape → [B*F,C,T] → multiple Conv1d → reshape

    Reshape-Free flow:
        [B,C,T,F] → multiple Conv2d(kernel=(K,1)) → [B,C,T,F]

    Args:
        channels: Number of input/output channels
        kernel_list: List of kernel sizes for multi-scale processing
        axis: Processing axis ('time' or 'freq')
        causal: If True and axis='time', use causal (left-only) padding
    """

    def __init__(
        self,
        channels: int,
        kernel_list: List[int] = [3, 5, 7, 11],
        axis: str = "time",
        causal: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_list = kernel_list
        self.causal = causal and (axis == "time")
        self.expand_ratio = len(kernel_list)
        mid_channel = channels * self.expand_ratio

        self.norm = ChannelLayerNorm2d(channels)
        self.proj_first = nn.Conv2d(channels, mid_channel, kernel_size=1)
        self.proj_last = nn.Conv2d(mid_channel, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Create conv layers for each kernel size
        for k in kernel_list:
            if axis == "time":
                if self.causal:
                    # Causal: left-only padding
                    attn_conv = nn.Sequential(
                        CausalConv2dTime(channels, channels, k, groups=channels),
                        nn.Conv2d(channels, channels, kernel_size=1),
                    )
                    main_conv = CausalConv2dTime(channels, channels, k, groups=channels)
                else:
                    # Non-causal: symmetric padding
                    attn_conv = nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=(k, 1),
                            padding=(k // 2, 0),
                            groups=channels,
                        ),
                        nn.Conv2d(channels, channels, kernel_size=1),
                    )
                    main_conv = nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(k, 1),
                        padding=(k // 2, 0),
                        groups=channels,
                    )
            else:
                # Freq axis: kernel (1, K) - always symmetric (non-causal)
                attn_conv = nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, k),
                        padding=(0, k // 2),
                        groups=channels,
                    ),
                    nn.Conv2d(channels, channels, kernel_size=1),
                )
                main_conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )

            setattr(self, f"attn_{k}", attn_conv)
            setattr(self, f"conv_{k}", main_conv)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Output tensor [B, C, T, F]
        """
        shortcut = x

        x = self.norm(x)
        x = self.proj_first(x)

        # Split into chunks for multi-scale processing
        chunks = x.chunk(self.expand_ratio, dim=1)
        outputs = []

        for i, k in enumerate(self.kernel_list):
            attn_module = getattr(self, f"attn_{k}")
            conv_module = getattr(self, f"conv_{k}")

            attn_out = attn_module(chunks[i])
            conv_out = conv_module(chunks[i])
            outputs.append(attn_out * conv_out)

        x = torch.cat(outputs, dim=1)
        x = self.proj_last(x) * self.scale + shortcut

        return x


class ReshapeFreeTSBlock(nn.Module):
    """
    Reshape-Free TS_BLOCK.

    Completely eliminates reshape operations by using Conv2d with axis-specific
    kernels. The time_stage uses kernel (K, 1) and freq_stage uses kernel (1, K).

    Original TS_BLOCK: 4 reshape operations per block
    Reshape-Free: 0 reshape operations

    Performance improvement: ~193x faster for the reshape + conv path

    Args:
        dense_channel: Number of channels
        time_block_num: Number of blocks in time stage
        freq_block_num: Number of blocks in freq stage
        time_dw_kernel_size: Kernel size for time stage depthwise conv
        time_block_kernel: Kernel sizes for time stage GPKFFN
        freq_block_kernel: Kernel sizes for freq stage GPKFFN
        causal: If True, use causal padding for time axis convolutions
    """

    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: List[int] = [3, 5, 7, 11],
        freq_block_kernel: List[int] = [3, 5, 7, 11],
        causal: bool = False,
    ):
        super().__init__()
        self.dense_channel = dense_channel
        self.causal = causal

        # Time stage: Conv2d with kernel (K, 1)
        time_blocks = []
        for _ in range(time_block_num):
            time_blocks.append(
                nn.Sequential(
                    ReshapeFreeCAB(
                        dense_channel,
                        kernel_size=time_dw_kernel_size,
                        axis="time",
                        causal=causal,
                    ),
                    ReshapeFreeGPKFFN(
                        dense_channel,
                        kernel_list=time_block_kernel,
                        axis="time",
                        causal=causal,
                    ),
                )
            )
        self.time_stage = nn.Sequential(*time_blocks)

        # Freq stage: Conv2d with kernel (1, K) - always non-causal
        freq_blocks = []
        for _ in range(freq_block_num):
            freq_blocks.append(
                nn.Sequential(
                    ReshapeFreeCAB(
                        dense_channel,
                        kernel_size=3,  # freq stage uses fixed kernel
                        axis="freq",
                        causal=False,  # freq is never causal
                    ),
                    ReshapeFreeGPKFFN(
                        dense_channel,
                        kernel_list=freq_block_kernel,
                        axis="freq",
                        causal=False,
                    ),
                )
            )
        self.freq_stage = nn.Sequential(*freq_blocks)

        # Residual scaling parameters [1, C, 1, 1] for 4D tensors
        self.beta_t = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))
        self.beta_f = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass WITHOUT any reshape operations.

        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Output tensor [B, C, T, F]
        """
        # Time stage processing (no reshape!)
        x = self.time_stage(x) + x * self.beta_t

        # Freq stage processing (no reshape!)
        x = self.freq_stage(x) + x * self.beta_f

        return x


__all__ = [
    "ChannelLayerNorm2d",
    "SimpleGate2d",
    "CausalConv2dTime",
    "ReshapeFreeCAB",
    "ReshapeFreeGPKFFN",
    "ReshapeFreeTSBlock",
]
