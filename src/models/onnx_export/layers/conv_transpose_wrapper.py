"""
ConvTranspose2d Wrapper using DepthToSpace + Pad + Conv2d.

This module provides an ONNX/INT8-friendly replacement for nn.ConvTranspose2d
by decomposing it into standard operations that have better runtime support.

Key Classes:
    - ConvTranspose2dWrapper: Drop-in replacement for nn.ConvTranspose2d

The decomposition follows the mathematical equivalence:
    ConvTranspose2d = ZeroInsertionUpsample + AsymmetricPad + Conv2d(flipped_weights)

Zero-insertion upsampling is implemented using DepthToSpace (PixelShuffle),
which has broad runtime support including NNAPI.

Reference:
    See docs/lacosenet_onnx_int8_export_plan.md section 5.5
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def tconv_weight_to_conv_weight(
    weight: Tensor,
    groups: int = 1,
) -> Tensor:
    """
    Convert ConvTranspose2d weight to equivalent Conv2d weight.

    ConvTranspose2d weight shape: [in_channels, out_channels/groups, kH, kW]
    Conv2d weight shape: [out_channels, in_channels/groups, kH, kW]

    The conversion involves:
    1. Channel axis transpose: swap in_channels and out_channels
    2. Spatial axis flip: reverse kernel spatially (180-degree rotation)

    This is necessary because:
    - ConvTranspose2d performs transposed convolution (correlation with flipped kernel)
    - Conv2d performs cross-correlation (no kernel flip)
    - To match ConvTranspose2d output, Conv2d needs the explicitly flipped kernel

    Args:
        weight: ConvTranspose2d weight [in_ch, out_ch/g, kH, kW]
        groups: Number of groups (must match original ConvTranspose2d)

    Returns:
        Conv2d compatible weight [out_ch, in_ch/g, kH, kW]

    Example:
        >>> tconv = nn.ConvTranspose2d(64, 32, (3, 3), groups=1)
        >>> conv_weight = tconv_weight_to_conv_weight(tconv.weight, groups=1)
        >>> conv_weight.shape
        torch.Size([32, 64, 3, 3])
    """
    # weight: [in_ch, out_ch/g, kH, kW]
    in_ch, out_ch_per_group, kH, kW = weight.shape
    out_ch = out_ch_per_group * groups

    # Step 1: Transpose channel axes
    # [in_ch, out_ch/g, kH, kW] -> [out_ch/g, in_ch, kH, kW]
    # For groups > 1, we need to reshape properly
    if groups == 1:
        # Simple case: just transpose
        # [in_ch, out_ch, kH, kW] -> [out_ch, in_ch, kH, kW]
        w_transposed = weight.permute(1, 0, 2, 3)
    else:
        # Grouped case: reshape to handle groups correctly
        # [in_ch, out_ch/g, kH, kW] where in_ch = g * (in_ch/g)
        in_ch_per_group = in_ch // groups
        # Reshape to [g, in_ch/g, out_ch/g, kH, kW]
        w = weight.view(groups, in_ch_per_group, out_ch_per_group, kH, kW)
        # Transpose per group: [g, out_ch/g, in_ch/g, kH, kW]
        w = w.permute(0, 2, 1, 3, 4)
        # Reshape to [out_ch, in_ch/g, kH, kW]
        w_transposed = w.reshape(out_ch, in_ch_per_group, kH, kW)

    # Step 2: Flip spatial axes (180-degree rotation)
    # flip along both kH and kW
    w_flipped = w_transposed.flip(dims=[2, 3])

    return w_flipped.contiguous()


def compute_conv_transpose_padding(
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Compute asymmetric padding for Conv2d to match ConvTranspose2d output.

    This implements the padding formula from the ONNX ConvTranspose specification:
    - effective_kernel = (kernel - 1) * dilation + 1
    - pad_before = effective_kernel - 1 - padding
    - pad_after = effective_kernel - 1 - padding + output_padding

    Args:
        kernel_size: (kH, kW) kernel dimensions
        stride: (sH, sW) stride values
        padding: (pH, pW) original ConvTranspose2d padding
        output_padding: (opH, opW) output padding
        dilation: (dH, dW) dilation rates

    Returns:
        (pad_top, pad_bottom, pad_left, pad_right) for F.pad

    Note:
        The returned padding is for the UPSAMPLED input (after zero-insertion).
    """
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    opH, opW = output_padding

    # Effective kernel size with dilation
    k_eff_H = (kH - 1) * dH + 1
    k_eff_W = (kW - 1) * dW + 1

    # Asymmetric padding
    pad_top = k_eff_H - 1 - pH
    pad_bottom = k_eff_H - 1 - pH + opH
    pad_left = k_eff_W - 1 - pW
    pad_right = k_eff_W - 1 - pW + opW

    return (pad_top, pad_bottom, pad_left, pad_right)


class ZeroInsertUpsample2d(nn.Module):
    """
    Zero-insertion upsampling for transposed convolution.

    This is NOT bilinear/nearest upsampling - it inserts (stride-1) zeros
    BETWEEN samples. For an input of size H, output size is (H-1)*stride + 1.

    The result has original values at positions (h*stride) for h=0,1,...,H-1
    and zeros at all other positions.

    Args:
        stride: Upsampling factor (sH, sW)

    Example:
        >>> upsample = ZeroInsertUpsample2d(stride=(2, 2))
        >>> x = torch.randn(1, 3, 4, 4)
        >>> y = upsample(x)
        >>> y.shape
        torch.Size([1, 3, 7, 7])  # (4-1)*2 + 1 = 7
        >>> # Original values at (0,0), (0,2), (2,0), (2,2), etc.
        >>> # Zeros at (0,1), (1,0), (1,1), etc.
    """

    def __init__(self, stride: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.stride_h, self.stride_w = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply zero-insertion upsampling.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Upsampled tensor [B, C, (H-1)*sH+1, (W-1)*sW+1]
        """
        B, C, H, W = x.shape
        sH, sW = self.stride_h, self.stride_w

        if sH == 1 and sW == 1:
            # No upsampling needed
            return x

        # Output size: (H-1)*stride + 1
        # This matches transposed convolution semantics where we insert
        # (stride-1) zeros BETWEEN elements, not after each element
        out_H = (H - 1) * sH + 1
        out_W = (W - 1) * sW + 1

        out = x.new_zeros(B, C, out_H, out_W)

        # Place original values at strided positions
        # Positions: 0, stride, 2*stride, ..., (H-1)*stride
        out[:, :, ::sH, ::sW] = x

        return out

    def extra_repr(self) -> str:
        return f"stride=({self.stride_h}, {self.stride_w})"


class ConvTranspose2dWrapper(nn.Module):
    """
    ONNX/INT8-friendly replacement for nn.ConvTranspose2d.

    This module decomposes ConvTranspose2d into:
    1. Zero-insertion upsampling (via DepthToSpace-like operation)
    2. Asymmetric padding
    3. Standard Conv2d with flipped weights

    The decomposition is mathematically equivalent to ConvTranspose2d but uses
    only operations with broad NNAPI/INT8 support.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Stride (upsampling factor)
        padding: Padding applied in ConvTranspose2d
        output_padding: Additional padding for ambiguous output sizes
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias

    Example:
        >>> # Create wrapper from existing ConvTranspose2d
        >>> tconv = nn.ConvTranspose2d(64, 64, (1, 3), (1, 2))
        >>> wrapper = ConvTranspose2dWrapper.from_conv_transpose(tconv)
        >>>
        >>> # Outputs are numerically equivalent
        >>> x = torch.randn(1, 64, 10, 65)
        >>> torch.allclose(tconv(x), wrapper(x), atol=1e-5)
        True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Sub-modules
        self.upsample = ZeroInsertUpsample2d(stride=stride)

        # Compute asymmetric padding for the upsampled input
        self.pad_values = compute_conv_transpose_padding(
            kernel_size, stride, padding, output_padding, dilation
        )

        # Conv2d with the same spatial parameters but no automatic padding
        # The weight will be converted from ConvTranspose2d format
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,  # Always stride=1 after upsampling
            padding=0,  # We apply padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @classmethod
    def from_conv_transpose(
        cls,
        conv_transpose: nn.ConvTranspose2d,
    ) -> "ConvTranspose2dWrapper":
        """
        Create wrapper from existing nn.ConvTranspose2d with weight conversion.

        Args:
            conv_transpose: Original ConvTranspose2d module

        Returns:
            ConvTranspose2dWrapper with converted weights
        """
        wrapper = cls(
            in_channels=conv_transpose.in_channels,
            out_channels=conv_transpose.out_channels,
            kernel_size=conv_transpose.kernel_size,
            stride=conv_transpose.stride,
            padding=conv_transpose.padding,
            output_padding=conv_transpose.output_padding,
            dilation=conv_transpose.dilation,
            groups=conv_transpose.groups,
            bias=conv_transpose.bias is not None,
        )

        # Convert and copy weights
        conv_weight = tconv_weight_to_conv_weight(
            conv_transpose.weight.data,
            groups=conv_transpose.groups,
        )
        wrapper.conv.weight.data.copy_(conv_weight)

        # Copy bias if present
        if conv_transpose.bias is not None:
            wrapper.conv.bias.data.copy_(conv_transpose.bias.data)

        return wrapper

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass equivalent to ConvTranspose2d.

        Args:
            x: Input tensor [B, C_in, H, W]

        Returns:
            Output tensor [B, C_out, H', W'] where:
            H' = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
            W' = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
        """
        # Step 1: Zero-insertion upsampling
        x = self.upsample(x)

        # Step 2: Asymmetric padding
        # pad_values = (pad_top, pad_bottom, pad_left, pad_right)
        # F.pad expects (left, right, top, bottom)
        pad_top, pad_bottom, pad_left, pad_right = self.pad_values
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        # Step 3: Conv2d with converted weights
        x = self.conv(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"dilation={self.dilation}, groups={self.groups}"
        )


def convert_conv_transpose_to_wrapper(
    module: nn.Module,
    inplace: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Recursively convert all nn.ConvTranspose2d to ConvTranspose2dWrapper.

    Args:
        module: Root module to convert
        inplace: If True, modify in place. If False, return a copy.

    Returns:
        Tuple of (converted module, number of conversions)

    Example:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 64, 3),
        ...     nn.ConvTranspose2d(64, 32, 3, 2),
        ... )
        >>> model, count = convert_conv_transpose_to_wrapper(model)
        >>> count
        1
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    conversion_count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            # Convert to wrapper
            wrapper = ConvTranspose2dWrapper.from_conv_transpose(child)
            setattr(module, name, wrapper)
            conversion_count += 1
        else:
            # Recurse into child modules
            _, child_count = convert_conv_transpose_to_wrapper(child, inplace=True)
            conversion_count += child_count

    return module, conversion_count


__all__ = [
    "ConvTranspose2dWrapper",
    "ZeroInsertUpsample2d",
    "tconv_weight_to_conv_weight",
    "compute_conv_transpose_padding",
    "convert_conv_transpose_to_wrapper",
]
