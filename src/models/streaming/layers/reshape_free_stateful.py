"""
Stateful Reshape-Free Layers for Streaming Inference.

This module provides stateful versions of reshape-free layers for streaming
inference with explicit state I/O (ONNX-exportable).

Key features:
    - State tensors as explicit inputs/outputs (no internal buffers)
    - Conv2d with axis-specific kernels (no reshape needed)
    - Compatible with batch_size=1 optimized inference

State shapes (B=1):
    - Time axis conv: [B, C, padding_t, F] = [1, 64, 2, 100]
    - SCA dwconv state: [B, C, sca_kernel_size-1, F] = [1, 64, 10, 100]

Usage:
    >>> from src.models.streaming.layers.reshape_free_stateful import (
    ...     StatefulReshapeFreeCAB
    ... )
    >>> cab = StatefulReshapeFreeCAB(64, axis='time')
    >>> state = cab.init_state(freq_size=100)
    >>> out, new_state = cab(x, state)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.streaming.layers.reshape_free import (
    AxisLayerNorm,
    ChannelLayerNorm2d,
    SimpleGate2d,
)


class StatefulReshapeFreeConv2d(nn.Module):
    """
    Stateful Conv2d for streaming inference.

    Maintains state buffer for causal (time-axis) convolutions.
    For freq-axis convolutions, state is typically not needed since
    frequency dimension is fully available.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size tuple (K_t, K_f)
        padding: Padding tuple (pad_t, pad_f)
        axis: Processing axis ('time' or 'freq')
        groups: Number of groups for grouped conv
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        axis: str = "time",
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.axis = axis
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # For time axis, we need left padding only (causal)
        # For freq axis, we use symmetric padding (non-causal)
        if axis == "time":
            self.padding_size = padding[0]  # Time padding
            self.state_dim = 2  # T dimension
            conv_padding = (0, padding[1])  # No time padding, keep freq padding
        else:
            self.padding_size = 0  # No state needed for freq
            self.state_dim = 3
            conv_padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=conv_padding,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        time_size: int = 40,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Initialize state tensor.

        For time axis: state shape [B, C, padding_t, F]
        For freq axis: no state needed (return empty)
        """
        if dtype is None:
            dtype = self.conv.weight.dtype
        if device is None:
            device = self.conv.weight.device

        if self.axis == "time" and self.padding_size > 0:
            return torch.zeros(
                batch_size,
                self.in_channels,
                self.padding_size,
                freq_size,
                device=device,
                dtype=dtype,
            )
        else:
            # No state needed for freq axis
            return torch.zeros(1, device=device, dtype=dtype)

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
            state: Previous state tensor
            state_frames: Number of frames (from start of input) to use for
                state update. If None, uses all T frames. This enables
                processing extended (lookahead) inputs while limiting state
                updates to the causal chunk.

        Returns:
            Tuple of (output, next_state)
        """
        if self.axis == "time" and self.padding_size > 0:
            # Concatenate state (past frames) with input
            x_padded = torch.cat([state, x], dim=2)

            # Compute output
            out = self.conv(x_padded)

            # Compute next state with optional gating
            effective_T = state_frames if state_frames is not None else x.shape[2]
            x_for_state = x[:, :, :effective_T, :]

            if effective_T >= self.padding_size:
                new_state = x_for_state[:, :, -self.padding_size :, :]
            else:
                keep = self.padding_size - effective_T
                old_part = state[:, :, -keep:, :]
                new_state = torch.cat([old_part, x_for_state], dim=2)

            return out, new_state
        else:
            # Freq axis: no state management
            out = self.conv(x)
            return out, state


class StatefulReshapeFreeCAB(nn.Module):
    """
    Stateful Reshape-Free Channel Attention Block.

    Combines:
    - Stateful depthwise Conv2d for temporal causality
    - Stateful depthwise Conv2d for SCA (causal local pooling replacement)

    State components:
    - dwconv_state: [B, 2C, padding, F] for causal depthwise conv
    - sca_dwconv_state: [B, C, sca_kernel_size-1, F] for causal SCA depthwise conv
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        axis: str = "time",
        sca_kernel_size: int = 11,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_size = kernel_size
        self.sca_kernel_size = sca_kernel_size
        dw_channel = channels * 2

        self.norm = ChannelLayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, dw_channel, kernel_size=1)

        # Stateful depthwise conv (causal: padding = kernel_size - 1)
        if axis == "time":
            causal_padding = kernel_size - 1  # Full left padding for causality
            self.dwconv = StatefulReshapeFreeConv2d(
                dw_channel,
                dw_channel,
                kernel_size=(kernel_size, 1),
                padding=(causal_padding, 0),
                axis="time",
                groups=dw_channel,
            )
            self.dwconv_padding = causal_padding
        else:
            # Freq axis: non-causal, no state needed
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dw_channel,
            )
            self.dwconv_padding = 0

        self.sg = SimpleGate2d()

        # SCA: depthwise conv (causal) + pointwise conv
        if axis == "time":
            sca_causal_padding = sca_kernel_size - 1
            self.sca_dwconv = StatefulReshapeFreeConv2d(
                channels,
                channels,
                kernel_size=(sca_kernel_size, 1),
                padding=(sca_causal_padding, 0),
                axis="time",
                groups=channels,
                bias=False,
            )
            self.sca_dwconv_padding = sca_causal_padding
        else:
            self.sca_dwconv = None
            self.sca_dwconv_padding = 0

        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Tensor]:
        """
        Initialize all states.

        Returns dict with:
        - 'dwconv': Conv state [B, 2C, padding, F]
        - 'sca_dwconv': SCA depthwise conv state [B, C, sca_kernel_size-1, F]
        """
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        states = {}

        # Dwconv state (only for time axis)
        if self.axis == "time" and self.dwconv_padding > 0:
            states["dwconv"] = torch.zeros(
                batch_size,
                self.channels * 2,
                self.dwconv_padding,
                freq_size,
                device=device,
                dtype=dtype,
            )
        else:
            states["dwconv"] = torch.zeros(1, device=device, dtype=dtype)

        # SCA depthwise conv state (only for time axis)
        if self.axis == "time" and self.sca_dwconv is not None:
            states["sca_dwconv"] = self.sca_dwconv.init_state(
                batch_size, freq_size, device=device, dtype=dtype,
            )
        else:
            states["sca_dwconv"] = torch.zeros(1, device=device, dtype=dtype)

        return states

    def forward(
        self,
        x: Tensor,
        state: Dict[str, Tensor],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Dict with 'dwconv' and 'sca_dwconv' states
            state_frames: Number of frames for state update gating.
                Passed through to dwconv and sca_dwconv.

        Returns:
            Tuple of (output, new_state_dict)
        """
        skip = x
        new_state = {}

        x = self.norm(x)
        x = self.pwconv1(x)

        # Stateful dwconv (time axis) or regular conv (freq axis)
        if self.axis == "time" and isinstance(self.dwconv, StatefulReshapeFreeConv2d):
            x, new_state["dwconv"] = self.dwconv(x, state["dwconv"], state_frames=state_frames)
        else:
            x = self.dwconv(x)
            new_state["dwconv"] = state["dwconv"]

        x = self.sg(x)

        # SCA: depthwise conv (causal) + pointwise conv
        if self.axis == "time" and self.sca_dwconv is not None:
            sca_out, new_state["sca_dwconv"] = self.sca_dwconv(
                x, state["sca_dwconv"], state_frames=state_frames
            )
            attn = self.sca_conv(sca_out)
        else:
            # Freq axis: simple mean (no state needed for frequency)
            attn = x.mean(dim=3, keepdim=True)
            attn = self.sca_conv(attn)
            new_state["sca_dwconv"] = state["sca_dwconv"]

        x = x * attn
        x = self.pwconv2(x)

        return skip + x * self.beta, new_state


class StatefulReshapeFreeGPKFFN(nn.Module):
    """
    Stateful Reshape-Free Group Prime Kernel FFN.

    Maintains state for each causal Conv2d in the multi-scale processing.
    """

    def __init__(
        self,
        channels: int,
        kernel_list: List[int] = [3, 5, 7, 11],
        axis: str = "time",
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_list = kernel_list
        self.expand_ratio = len(kernel_list)
        mid_channel = channels * self.expand_ratio

        self.norm = ChannelLayerNorm2d(channels)
        self.proj_first = nn.Conv2d(channels, mid_channel, kernel_size=1)
        self.proj_last = nn.Conv2d(mid_channel, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Create stateful conv layers for each kernel
        for k in kernel_list:
            if axis == "time":
                # Stateful for time axis (causal: padding = k - 1)
                causal_padding = k - 1
                attn_dw = StatefulReshapeFreeConv2d(
                    channels,
                    channels,
                    kernel_size=(k, 1),
                    padding=(causal_padding, 0),
                    axis="time",
                    groups=channels,
                )
                attn_pw = nn.Conv2d(channels, channels, kernel_size=1)
                main_conv = StatefulReshapeFreeConv2d(
                    channels,
                    channels,
                    kernel_size=(k, 1),
                    padding=(causal_padding, 0),
                    axis="time",
                    groups=channels,
                )
            else:
                # Non-stateful for freq axis
                attn_dw = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )
                attn_pw = nn.Conv2d(channels, channels, kernel_size=1)
                main_conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )

            setattr(self, f"attn_dw_{k}", attn_dw)
            setattr(self, f"attn_pw_{k}", attn_pw)
            setattr(self, f"conv_{k}", main_conv)

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Tensor]:
        """Initialize states for all causal convs."""
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        states = {}

        for k in self.kernel_list:
            causal_padding = k - 1  # Causal: full left padding
            if self.axis == "time" and causal_padding > 0:
                states[f"attn_dw_{k}"] = torch.zeros(
                    batch_size, self.channels, causal_padding, freq_size, device=device, dtype=dtype
                )
                states[f"conv_{k}"] = torch.zeros(
                    batch_size, self.channels, causal_padding, freq_size, device=device, dtype=dtype
                )
            else:
                states[f"attn_dw_{k}"] = torch.zeros(1, device=device, dtype=dtype)
                states[f"conv_{k}"] = torch.zeros(1, device=device, dtype=dtype)

        return states

    def forward(
        self,
        x: Tensor,
        state: Dict[str, Tensor],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Dict of state tensors for each causal conv
            state_frames: Number of frames for state update gating.

        Returns:
            Tuple of (output, new_state_dict)
        """
        shortcut = x
        new_state = {}

        x = self.norm(x)
        x = self.proj_first(x)

        chunks = x.chunk(self.expand_ratio, dim=1)
        outputs = []

        for i, k in enumerate(self.kernel_list):
            attn_dw = getattr(self, f"attn_dw_{k}")
            attn_pw = getattr(self, f"attn_pw_{k}")
            conv = getattr(self, f"conv_{k}")

            # Attention path
            if isinstance(attn_dw, StatefulReshapeFreeConv2d):
                attn_out, new_state[f"attn_dw_{k}"] = attn_dw(
                    chunks[i], state[f"attn_dw_{k}"], state_frames=state_frames
                )
            else:
                attn_out = attn_dw(chunks[i])
                new_state[f"attn_dw_{k}"] = state[f"attn_dw_{k}"]

            attn_out = attn_pw(attn_out)

            # Conv path
            if isinstance(conv, StatefulReshapeFreeConv2d):
                conv_out, new_state[f"conv_{k}"] = conv(
                    chunks[i], state[f"conv_{k}"], state_frames=state_frames
                )
            else:
                conv_out = conv(chunks[i])
                new_state[f"conv_{k}"] = state[f"conv_{k}"]

            outputs.append(attn_out * conv_out)

        x = torch.cat(outputs, dim=1)
        x = self.proj_last(x) * self.scale + shortcut

        return x, new_state


class StatefulReshapeFreeTSBlock(nn.Module):
    """
    Stateful Reshape-Free TS_BLOCK for streaming inference.

    Combines all stateful components:
    - Time stage: Stateful CAB + GPKFFN (causal)
    - Freq stage: Non-stateful CAB + GPKFFN (full context)
    """

    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: List[int] = [3, 5, 7, 11],
        freq_block_kernel: List[int] = [3, 5, 7, 11],
        sca_kernel_size: int = 11,
    ):
        super().__init__()
        self.dense_channel = dense_channel
        self.time_block_num = time_block_num
        self.freq_block_num = freq_block_num

        # Time stage (stateful)
        self.time_cabs = nn.ModuleList()
        self.time_gpkffns = nn.ModuleList()
        for _ in range(time_block_num):
            self.time_cabs.append(
                StatefulReshapeFreeCAB(
                    dense_channel,
                    kernel_size=time_dw_kernel_size,
                    axis="time",
                    sca_kernel_size=sca_kernel_size,
                )
            )
            self.time_gpkffns.append(
                StatefulReshapeFreeGPKFFN(
                    dense_channel,
                    kernel_list=time_block_kernel,
                    axis="time",
                )
            )

        # Freq stage (non-stateful, uses full freq context)
        from src.models.streaming.layers.reshape_free import (
            ReshapeFreeCAB,
            ReshapeFreeGPKFFN,
        )

        freq_blocks = []
        for _ in range(freq_block_num):
            freq_blocks.append(
                nn.Sequential(
                    ReshapeFreeCAB(dense_channel, kernel_size=3, axis="freq"),
                    ReshapeFreeGPKFFN(dense_channel, kernel_list=freq_block_kernel, axis="freq"),
                )
            )
        self.freq_stage = nn.Sequential(*freq_blocks)

        self.beta_t = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))
        self.beta_f = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, Tensor]]:
        """Initialize all states for time stage."""
        states = []

        for i in range(self.time_block_num):
            cab_state = self.time_cabs[i].init_state(batch_size, freq_size, device, dtype)
            gpkffn_state = self.time_gpkffns[i].init_state(batch_size, freq_size, device, dtype)
            states.append({"cab": cab_state, "gpkffn": gpkffn_state})

        return states

    def forward(
        self,
        x: Tensor,
        state: List[Dict[str, Tensor]],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: List of state dicts for each time block
            state_frames: Number of frames for state update gating.

        Returns:
            Tuple of (output, new_states)
        """
        new_states = []
        residual = x

        # Time stage (stateful)
        for i in range(self.time_block_num):
            x, cab_state = self.time_cabs[i](x, state[i]["cab"], state_frames=state_frames)
            x, gpkffn_state = self.time_gpkffns[i](x, state[i]["gpkffn"], state_frames=state_frames)
            new_states.append({"cab": cab_state, "gpkffn": gpkffn_state})

        x = x + residual * self.beta_t

        # Freq stage (non-stateful)
        x = self.freq_stage(x) + x * self.beta_f

        return x, new_states


__all__ = [
    "StatefulReshapeFreeConv2d",
    "StatefulReshapeFreeCAB",
    "StatefulReshapeFreeGPKFFN",
    "StatefulReshapeFreeTSBlock",
]
