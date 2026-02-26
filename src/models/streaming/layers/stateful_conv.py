"""
Stateful Convolution Layers for Streaming Inference.

This module provides stateful versions of CausalConv1d and AsymmetricConv2d
that maintain activation buffers for seamless chunk-by-chunk processing,
eliminating zero-padding discontinuity at chunk boundaries.

Key Classes:
    - StatefulCausalConv1d: Stateful version of CausalConv1d for 1D temporal data
    - StatefulAsymmetricConv2d: Stateful version of AsymmetricConv2d for 2D spectrogram data
    - StatefulCausalConv2d: Stateful version of CausalConv2d

For conversion utilities, see:
    src.models.streaming.converters.conv_converter

Example:
    >>> from src.models.streaming.layers import StatefulCausalConv1d
    >>> from src.models.backbone import CausalConv1d
    >>>
    >>> # Create original conv and convert to stateful
    >>> original = CausalConv1d(64, 64, kernel_size=3, padding=1)
    >>> stateful = StatefulCausalConv1d.from_causal_conv(original)
    >>>
    >>> # Enable streaming mode
    >>> stateful.set_streaming(True)
    >>>
    >>> # Process chunks
    >>> for chunk in audio_chunks:
    ...     output = stateful(chunk)

Reference:
    See docs/STATEFUL_STREAMING_TECH_SPEC.md for detailed technical specification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn  # Use 'fn' to avoid conflict with 'F' (frequency) variable

# Import from canonical locations
from src.models.streaming.utils import (
    get_state_frames_context,
    check_batch_size_change,
    check_streaming_allowed,
    sync_state_device_dtype,
)

if TYPE_CHECKING:
    from src.models.backbone import AsymmetricConv2d, CausalConv1d

logger = logging.getLogger(__name__)


class StatefulLayerMixin:
    """
    Mixin providing the common stateful streaming layer interface.

    Provides reset_state(), set_streaming(), and state property.
    Subclasses must set ``self._state = None`` and ``self._streaming = False``
    in their own ``__init__`` (after ``super().__init__()``).

    Requires the host class to also inherit from ``nn.Module``
    (uses ``self.training`` from Module).
    """

    _state: Optional[torch.Tensor]
    _streaming: bool

    def reset_state(self) -> None:
        """Reset state for new utterance."""
        self._state = None

    def set_streaming(self, streaming: bool) -> None:
        """
        Enable/disable streaming mode with safety checks.

        Args:
            streaming: True to enable streaming mode

        Raises:
            RuntimeError: If streaming is enabled during training mode
        """
        if streaming:
            check_streaming_allowed(self.training, type(self).__name__)

        if self._streaming and not streaming:
            self.reset_state()

        self._streaming = streaming

    @property
    def state(self) -> Optional[torch.Tensor]:
        """Get current state."""
        return self._state


class StatefulCausalConv1d(StatefulLayerMixin, nn.Module):
    """
    Stateful version of CausalConv1d for streaming inference.

    Maintains a state buffer containing the last `padding_size` frames
    from the previous chunk, using it as left-padding instead of zeros.

    This eliminates the zero-padding discontinuity that occurs at chunk
    boundaries during streaming inference, making the output equivalent
    to processing the full sequence at once.

    Attributes:
        padding_size: Number of frames to pad on the left
        conv: Underlying Conv1d layer
        _state: Internal buffer for previous activation [B, C, padding_size]
        _streaming: Whether streaming mode is enabled

    Note:
        - State is only maintained when streaming=True
        - State is automatically reset when streaming mode is disabled
        - First chunk always uses zero padding (no previous context)
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
        # Match original CausalConv1d behavior where padding is doubled
        self.padding_size = padding * 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # No automatic padding
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # State buffer (initialized on first forward)
        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_causal_conv(cls, causal_conv: "CausalConv1d") -> "StatefulCausalConv1d":
        """
        Convert existing CausalConv1d to StatefulCausalConv1d.

        Uses safe weight copy approach instead of reference sharing to ensure:
        - Proper state_dict save/load compatibility
        - TorchScript/ONNX export compatibility
        - Independence from original module

        Args:
            causal_conv: Original CausalConv1d instance

        Returns:
            New StatefulCausalConv1d with copied weights

        Note:
            causal_conv.padding is already multiplied by 2 in CausalConv1d.__init__
            (see backbone.py line 18: self.padding = padding * 2)
        """
        orig_conv = causal_conv.conv

        # Create new instance with full __init__ (safe approach)
        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size[0],
            padding=causal_conv.padding // 2,  # Restore original padding value
            stride=orig_conv.stride[0],
            dilation=orig_conv.dilation[0],
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )

        # Copy weights (not share reference)
        instance.conv.load_state_dict(orig_conv.state_dict())

        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional stateful streaming.

        Args:
            x: Input tensor [B, C, T]
            state_frames: Number of frames to use for state update.
                If None, check for context-based value, then use all frames (T).
                If specified, only use first state_frames frames for state update.

        Returns:
            Output tensor [B, C_out, T]
        """
        if not self._streaming:
            # Non-streaming: original behavior (zero padding)
            x = fn.pad(x, [self.padding_size, 0])
            return self.conv(x)

        # Streaming mode: use state as left padding
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Batch size change detection
        if check_batch_size_change(self._state, B, "StatefulCausalConv1d"):
            self.reset_state()

        # Device/dtype handling
        self._state = sync_state_device_dtype(self._state, device, dtype)

        if self._state is None:
            # First chunk: use zero padding
            left_pad = torch.zeros(B, C, self.padding_size, device=device, dtype=dtype)
        else:
            # Subsequent chunks: use saved state
            left_pad = self._state

        # Concatenate: [state/zeros | current_input]
        x_padded = torch.cat([left_pad, x], dim=2)

        # Update state: save last padding_size frames of INPUT (not output)
        # This ensures continuity at the input level
        # Check for context-based state_frames if not explicitly provided
        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T]

        if effective_T >= self.padding_size:
            self._state = x_for_state[:, :, -self.padding_size :].clone().detach()
        else:
            # Input shorter than padding: combine old state and new input
            keep = self.padding_size - effective_T
            if self._state is not None:
                old_part = self._state[:, :, -keep:]
            else:
                old_part = torch.zeros(B, C, keep, device=device, dtype=dtype)
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[StatefulCausalConv1d] streaming={self._streaming}, "
                f"state={'initialized' if self._state is not None else 'None'}, "
                f"input_shape={list(x.shape)}, state_frames={effective_T}"
            )

        return self.conv(x_padded)


class StatefulAsymmetricConv2d(StatefulLayerMixin, nn.Module):
    """
    Stateful version of AsymmetricConv2d for streaming inference.

    Handles asymmetric padding on the time axis while maintaining
    symmetric padding on the frequency axis.

    State is maintained only for the left (past) time padding.
    Right (future) time padding is zero-padded by default. For models with
    asymmetric padding that require future context, use LaCoSENet
    which provides real lookahead frames through input buffering.

    Attributes:
        time_padding_left: Left padding size on time axis
        time_padding_right: Right padding size on time axis
        freq_padding: Symmetric padding on frequency axis
        conv: Underlying Conv2d layer
        _state: Buffer for previous activation [B, C, time_pad_left, F_padded]
        _streaming: Whether streaming mode is enabled

    Note:
        - For fully causal models (padding_ratio=(1.0, 0.0)): streaming output
          matches full-sequence output exactly.
        - For asymmetric models (right_ratio > 0): use with BufferedLaCoSENet
          to provide real future context. The wrapper provides extended input that
          includes lookahead frames, so right padding effectively has real data.
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

        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_asymmetric_conv(
        cls, asym_conv: "AsymmetricConv2d"
    ) -> "StatefulAsymmetricConv2d":
        """
        Convert existing AsymmetricConv2d to StatefulAsymmetricConv2d.

        Uses safe weight copy approach instead of reference sharing to ensure:
        - Proper state_dict save/load compatibility
        - TorchScript/ONNX export compatibility
        - Independence from original module

        Args:
            asym_conv: Original AsymmetricConv2d instance

        Returns:
            New StatefulAsymmetricConv2d with copied weights

        Note:
            AsymmetricConv2d stores padding as:
            self.padding = (freq_padding, freq_padding, time_padding_left, time_padding_right)
            (see backbone.py line 113)

        IMPORTANT:
            stride and dilation are preserved as-is (tuple for 2D conv).
            Do NOT extract single axis - both (time, freq) axes must match original.
        """
        orig_conv = asym_conv.conv

        # Recover original padding values from AsymmetricConv2d
        # time_padding_left + time_padding_right = padding[0] * 2 (total time padding)
        total_time_padding = asym_conv.time_padding_left + asym_conv.time_padding_right
        original_time_padding = total_time_padding // 2
        original_freq_padding = asym_conv.padding[0]  # freq_padding stored directly

        # Create new instance with full __init__ (safe approach)
        # NOTE: stride and dilation must preserve both axes (time, freq) for 2D conv
        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,  # tuple: (kernel_time, kernel_freq)
            padding=(original_time_padding, original_freq_padding),
            padding_ratio=asym_conv.padding_ratio,
            stride=orig_conv.stride,  # tuple: (stride_time, stride_freq)
            dilation=orig_conv.dilation,  # tuple: (dilation_time, dilation_freq)
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )

        # Copy weights (not share reference)
        instance.conv.load_state_dict(orig_conv.state_dict())

        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional stateful streaming.

        Args:
            x: Input tensor [B, C, T, F] where F is frequency dimension
            state_frames: Number of frames to use for state update.
                If None, use all frames (T). If specified, only use first
                state_frames frames for state update. This is useful when
                processing extended input with lookahead context - we don't
                want the lookahead frames to corrupt the state buffer.

        Returns:
            Output tensor [B, C_out, T, F]

        Note:
            - Frequency padding is always symmetric (no temporal dependency)
            - State is saved AFTER frequency padding is applied
            - When state_frames < T, the lookahead portion is processed but
              does not affect state for the next chunk
        """
        B, C, T, freq_dim = x.shape
        device = x.device
        dtype = x.dtype

        if not self._streaming:
            # Non-streaming: original behavior
            pad = (
                self.freq_padding,
                self.freq_padding,
                self.time_padding_left,
                self.time_padding_right,
            )
            x = fn.pad(x, pad)
            return self.conv(x)

        # Streaming mode
        # Batch size change detection
        if check_batch_size_change(self._state, B, "StatefulAsymmetricConv2d"):
            self.reset_state()

        # Device/dtype handling
        self._state = sync_state_device_dtype(self._state, device, dtype)

        # 1. Frequency padding (always symmetric, always zero - no temporal dependency)
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding with state
        if self._state is None:
            left_pad = torch.zeros(
                B, C, self.time_padding_left, freq_padded, device=device, dtype=dtype
            )
        else:
            left_pad = self._state

        # Right padding is always zero (future frames not available in streaming)
        right_pad = torch.zeros(
            B, C, self.time_padding_right, freq_padded, device=device, dtype=dtype
        )

        # Concatenate: [left_state | current | right_zeros]
        x_padded = torch.cat([left_pad, x, right_pad], dim=2)

        # 3. Update state: save last time_padding_left frames
        # State includes frequency padding (freq_padded size)
        # Check for context-based state_frames if not explicitly provided
        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]

        if effective_T >= self.time_padding_left:
            self._state = x_for_state[:, :, -self.time_padding_left :, :].clone().detach()
        else:
            keep = self.time_padding_left - effective_T
            if self._state is not None:
                old_part = self._state[:, :, -keep:, :]
            else:
                old_part = torch.zeros(
                    B, C, keep, freq_padded, device=device, dtype=dtype
                )
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[StatefulAsymmetricConv2d] streaming={self._streaming}, "
                f"state={'initialized' if self._state is not None else 'None'}, "
                f"input_shape={list(x.shape)}, state_frames={effective_T}"
            )

        return self.conv(x_padded)


class StatefulCausalConv2d(StatefulLayerMixin, nn.Module):
    """
    Stateful version of CausalConv2d for streaming inference.

    Similar to StatefulAsymmetricConv2d but with fixed causal padding
    (padding_ratio always (1.0, 0.0)).

    Attributes:
        time_padding: Left padding size on time axis (right is always 0)
        freq_padding: Symmetric padding on frequency axis
        conv: Underlying Conv2d layer
        _state: Buffer for previous activation [B, C, time_padding, F_padded]
        _streaming: Whether streaming mode is enabled
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

        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_causal_conv2d(cls, causal_conv: "CausalConv2d") -> "StatefulCausalConv2d":
        """
        Convert existing CausalConv2d to StatefulCausalConv2d.

        Args:
            causal_conv: Original CausalConv2d instance

        Returns:
            New StatefulCausalConv2d with copied weights
        """
        from src.models.backbone import CausalConv2d

        orig_conv = causal_conv.conv

        # CausalConv2d stores padding as (freq, freq, time, 0)
        # time_padding = padding[0] * 2 (doubled in __init__)
        time_padding_total = causal_conv.padding[2]  # time padding (left)
        freq_padding = causal_conv.padding[0]  # freq padding

        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            padding=(time_padding_total // 2, freq_padding),
            stride=orig_conv.stride,
            dilation=orig_conv.dilation,
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )

        instance.conv.load_state_dict(orig_conv.state_dict())

        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional stateful streaming.

        Args:
            x: Input tensor [B, C, T, F]
            state_frames: Number of frames to use for state update.
                If None, use all frames (T). If specified, only use first
                state_frames frames for state update.

        Returns:
            Output tensor [B, C_out, T, F]
        """
        B, C, T, freq_dim = x.shape
        device = x.device
        dtype = x.dtype

        if not self._streaming:
            # Non-streaming: original behavior
            pad = (self.freq_padding, self.freq_padding, self.time_padding, 0)
            x = fn.pad(x, pad)
            return self.conv(x)

        # Streaming mode
        if check_batch_size_change(self._state, B, "StatefulCausalConv2d"):
            self.reset_state()

        self._state = sync_state_device_dtype(self._state, device, dtype)

        # 1. Frequency padding
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding with state
        if self._state is None:
            left_pad = torch.zeros(
                B, C, self.time_padding, freq_padded, device=device, dtype=dtype
            )
        else:
            left_pad = self._state

        # Concatenate: [left_state | current]
        x_padded = torch.cat([left_pad, x], dim=2)

        # 3. Update state
        # Check for context-based state_frames if not explicitly provided
        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]

        if effective_T >= self.time_padding:
            self._state = x_for_state[:, :, -self.time_padding :, :].clone().detach()
        else:
            keep = self.time_padding - effective_T
            if self._state is not None:
                old_part = self._state[:, :, -keep:, :]
            else:
                old_part = torch.zeros(
                    B, C, keep, freq_padded, device=device, dtype=dtype
                )
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        return self.conv(x_padded)


__all__ = [
    "StatefulLayerMixin",
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
]
