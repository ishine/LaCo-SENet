"""
Functional Stateful Layers for ONNX Export.

This module provides ONNX-exportable versions of stateful convolutions
that use explicit state I/O instead of internal buffers.

Key differences from stateful_conv.py:
1. State is passed as input and returned as output (no internal _state)
2. No clone().detach() - state tensors are part of the computation graph
3. No StateFramesContext - state_frames passed explicitly
4. Designed for step-graph ONNX export

Example:
    >>> conv = FunctionalStatefulConv1d(64, 64, kernel_size=3, padding=1)
    >>> state = conv.init_state(batch_size=1, device='cuda')
    >>> for chunk in audio_chunks:
    ...     output, state = conv(chunk, state)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor


class FunctionalStatefulConv1d(nn.Module):
    """
    ONNX-exportable stateful CausalConv1d.

    State is explicitly passed in and out, enabling ONNX export with
    state as graph I/O.

    Forward signature:
        y, next_state = forward(x, state, state_frames=None)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: Original CausalConv1d padding value (will be doubled)
        stride: Convolution stride
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Causal padding: all on the left side
        self.padding_size = padding * 2
        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Initialize state tensor with zeros.

        Args:
            batch_size: Batch size
            device: Device for the state tensor
            dtype: Data type for the state tensor

        Returns:
            Zero-initialized state [B, C, padding_size]
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.in_channels, self.padding_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T]
            state: Previous state [B, C, padding_size]
            state_frames: Number of frames to use for state update.
                If None, uses all frames (T).

        Returns:
            Tuple of (output, next_state):
            - output: [B, C_out, T]
            - next_state: [B, C, padding_size]
        """
        B, C, T = x.shape

        # Concatenate: [state | current_input]
        x_padded = torch.cat([state, x], dim=2)

        # Compute output
        output = self.conv(x_padded)

        # Compute next state
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T]

        if effective_T >= self.padding_size:
            # Take last padding_size frames from input
            next_state = x_for_state[:, :, -self.padding_size:]
        else:
            # Input shorter than padding: combine old state and new input
            keep = self.padding_size - effective_T
            old_part = state[:, :, -keep:]
            next_state = torch.cat([old_part, x_for_state], dim=2)

        return output, next_state


class FunctionalStatefulConv2d(nn.Module):
    """
    ONNX-exportable stateful AsymmetricConv2d.

    State is explicitly passed in and out, enabling ONNX export with
    state as graph I/O.

    Forward signature:
        y, next_state = forward(x, state, state_frames=None)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: (time, freq) kernel size tuple
        padding: (time_padding, freq_padding) tuple
        padding_ratio: (left_ratio, right_ratio) for asymmetric time padding
        stride: (time, freq) stride tuple
        dilation: (time, freq) dilation tuple
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Calculate asymmetric time padding
        time_padding_total = padding[0] * 2
        freq_padding = padding[1]
        left_ratio, right_ratio = padding_ratio

        self.time_padding_left = round(time_padding_total * left_ratio)
        self.time_padding_right = round(time_padding_total * right_ratio)

        # Ensure total is preserved
        if self.time_padding_left + self.time_padding_right != time_padding_total:
            self.time_padding_right = time_padding_total - self.time_padding_left

        self.freq_padding = freq_padding
        self.padding_ratio = padding_ratio
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 257,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Initialize state tensor with zeros.

        Args:
            batch_size: Batch size
            freq_size: Frequency dimension (after freq_padding applied)
            device: Device for the state tensor
            dtype: Data type for the state tensor

        Returns:
            Zero-initialized state [B, C, time_padding_left, freq_padded]
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(
            batch_size, self.in_channels, self.time_padding_left, freq_padded,
            device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Previous state [B, C, time_padding_left, F_padded]
            state_frames: Number of frames to use for state update.
                If None, uses all frames (T).

        Returns:
            Tuple of (output, next_state):
            - output: [B, C_out, T, F]
            - next_state: [B, C, time_padding_left, F_padded]
        """
        B, C, T, F = x.shape

        # 1. Frequency padding (always symmetric)
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding with state
        # Right padding is always zero (future frames)
        right_pad = x.new_zeros(B, C, self.time_padding_right, freq_padded)

        # Concatenate: [left_state | current | right_zeros]
        x_padded = torch.cat([state, x, right_pad], dim=2)

        # 3. Compute output
        output = self.conv(x_padded)

        # 4. Compute next state
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]

        if effective_T >= self.time_padding_left:
            next_state = x_for_state[:, :, -self.time_padding_left:, :]
        else:
            keep = self.time_padding_left - effective_T
            old_part = state[:, :, -keep:, :]
            next_state = torch.cat([old_part, x_for_state], dim=2)

        return output, next_state


class FunctionalStatefulCausalConv2d(nn.Module):
    """
    ONNX-exportable stateful CausalConv2d (fully causal version).

    This is a simplified version of FunctionalStatefulConv2d with
    padding_ratio fixed to (1.0, 0.0).

    Forward signature:
        y, next_state = forward(x, state, state_frames=None)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Causal: all time padding on left
        self.time_padding = padding[0] * 2
        self.freq_padding = padding[1]
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 257,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Initialize state tensor with zeros."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(
            batch_size, self.in_channels, self.time_padding, freq_padded,
            device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with explicit state I/O."""
        B, C, T, F = x.shape

        # 1. Frequency padding
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding with state (causal: no right padding)
        x_padded = torch.cat([state, x], dim=2)

        # 3. Compute output
        output = self.conv(x_padded)

        # 4. Compute next state
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]

        if effective_T >= self.time_padding:
            next_state = x_for_state[:, :, -self.time_padding:, :]
        else:
            keep = self.time_padding - effective_T
            old_part = state[:, :, -keep:, :]
            next_state = torch.cat([old_part, x_for_state], dim=2)

        return output, next_state


def convert_to_functional(
    stateful_conv: nn.Module,
) -> nn.Module:
    """
    Convert StatefulConv to FunctionalStateful version.

    Args:
        stateful_conv: Original StatefulCausalConv1d, StatefulAsymmetricConv2d,
            or StatefulCausalConv2d

    Returns:
        Corresponding FunctionalStateful version with copied weights
    """
    from src.models.streaming.layers.stateful_conv import (
        StatefulAsymmetricConv2d,
        StatefulCausalConv1d,
        StatefulCausalConv2d,
    )

    if isinstance(stateful_conv, StatefulCausalConv1d):
        orig_conv = stateful_conv.conv
        functional = FunctionalStatefulConv1d(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size[0],
            padding=stateful_conv.padding_size // 2,
            stride=orig_conv.stride[0],
            dilation=orig_conv.dilation[0],
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        functional.conv.load_state_dict(orig_conv.state_dict())
        return functional

    elif isinstance(stateful_conv, StatefulAsymmetricConv2d):
        orig_conv = stateful_conv.conv
        total_time = stateful_conv.time_padding_left + stateful_conv.time_padding_right
        functional = FunctionalStatefulConv2d(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            padding=(total_time // 2, stateful_conv.freq_padding),
            padding_ratio=stateful_conv.padding_ratio,
            stride=orig_conv.stride,
            dilation=orig_conv.dilation,
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        functional.conv.load_state_dict(orig_conv.state_dict())
        return functional

    elif isinstance(stateful_conv, StatefulCausalConv2d):
        orig_conv = stateful_conv.conv
        functional = FunctionalStatefulCausalConv2d(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            padding=(stateful_conv.time_padding // 2, stateful_conv.freq_padding),
            stride=orig_conv.stride,
            dilation=orig_conv.dilation,
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        functional.conv.load_state_dict(orig_conv.state_dict())
        return functional

    else:
        raise TypeError(f"Unsupported type: {type(stateful_conv)}")


__all__ = [
    "FunctionalStatefulConv1d",
    "FunctionalStatefulConv2d",
    "FunctionalStatefulCausalConv2d",
    "convert_to_functional",
]
