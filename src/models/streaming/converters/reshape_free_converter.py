"""
Reshape-Free Model Converter.

This module provides utilities to convert existing Backbone models to
Reshape-Free versions that eliminate reshape operations for batch_size=1
optimized inference.

Key conversions:
    - Conv1d → Conv2d with axis-specific kernels
    - LayerNorm1d → AxisLayerNorm
    - TS_BLOCK → ReshapeFreeTSBlock (with weight transfer)

Usage:
    >>> from src.models.streaming.converters.reshape_free_converter import (
    ...     convert_backbone_to_reshape_free
    ... )
    >>> model = load_backbone_checkpoint(...)
    >>> rf_model = convert_backbone_to_reshape_free(model)

Reference:
    See android/docs/BATCH1_OPTIMIZATION_STRATEGY.md for detailed analysis.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.streaming.layers.reshape_free import (
    AxisLayerNorm,
    ReshapeFreeCAB,
    ReshapeFreeGPKFFN,
    ReshapeFreeTSBlock,
)

logger = logging.getLogger(__name__)


def convert_conv1d_to_conv2d(
    conv1d: nn.Conv1d,
    axis: str,
) -> nn.Conv2d:
    """
    Convert Conv1d to Conv2d with axis-specific kernel.

    For time axis: Conv1d(K) → Conv2d((K, 1))
    For freq axis: Conv1d(K) → Conv2d((1, K))

    Args:
        conv1d: Original Conv1d layer
        axis: Processing axis ('time' or 'freq')

    Returns:
        Conv2d layer with equivalent computation
    """
    # Extract Conv1d parameters
    in_channels = conv1d.in_channels
    out_channels = conv1d.out_channels
    kernel_size = conv1d.kernel_size[0]
    stride = conv1d.stride[0]
    padding = conv1d.padding[0]
    dilation = conv1d.dilation[0]
    groups = conv1d.groups
    has_bias = conv1d.bias is not None

    # Create Conv2d with axis-specific kernel
    if axis == "time":
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=groups,
            bias=has_bias,
        )
        # Weight shape: [C_out, C_in/g, K] → [C_out, C_in/g, K, 1]
        conv2d.weight.data = conv1d.weight.data.unsqueeze(-1)
    else:
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=has_bias,
        )
        # Weight shape: [C_out, C_in/g, K] → [C_out, C_in/g, 1, K]
        conv2d.weight.data = conv1d.weight.data.unsqueeze(2)

    if has_bias:
        conv2d.bias.data = conv1d.bias.data.clone()

    return conv2d


def convert_layernorm1d_to_axis(
    ln1d: nn.Module,
    axis: str,
) -> AxisLayerNorm:
    """
    Convert LayerNorm1d to AxisLayerNorm.

    Args:
        ln1d: Original LayerNorm1d layer (custom or nn.LayerNorm)
        axis: Processing axis ('time' or 'freq')

    Returns:
        AxisLayerNorm with transferred weights
    """
    # Get channels from weight shape
    if hasattr(ln1d, "weight"):
        channels = ln1d.weight.shape[0]
    else:
        raise ValueError(f"Cannot determine channels from {type(ln1d)}")

    eps = getattr(ln1d, "eps", 1e-6)

    axis_ln = AxisLayerNorm(channels=channels, axis=axis, eps=eps)

    # Transfer weights: [C] → [C, 1, 1]
    if hasattr(ln1d, "weight") and ln1d.weight is not None:
        axis_ln.weight.data = ln1d.weight.data.view(channels, 1, 1)
    if hasattr(ln1d, "bias") and ln1d.bias is not None:
        axis_ln.bias.data = ln1d.bias.data.view(channels, 1, 1)

    return axis_ln


def _convert_layernorm1d_to_channel(
    ln1d: nn.Module,
) -> "ChannelLayerNorm2d":
    """
    Convert LayerNorm1d to ChannelLayerNorm2d.

    ChannelLayerNorm2d normalizes over the channel dimension (dim=1) in 4D,
    which is the correct 4D equivalent of LayerNorm1d that normalizes
    over channels (dim=1) in 3D.

    Args:
        ln1d: Original LayerNorm1d layer

    Returns:
        ChannelLayerNorm2d with transferred weights
    """
    from src.models.streaming.layers.reshape_free import ChannelLayerNorm2d

    if hasattr(ln1d, "weight"):
        channels = ln1d.weight.shape[0]
    else:
        raise ValueError(f"Cannot determine channels from {type(ln1d)}")

    eps = getattr(ln1d, "eps", 1e-6)
    ch_ln = ChannelLayerNorm2d(channels=channels, eps=eps)

    # Transfer weights: [C] → [1, C, 1, 1]
    if hasattr(ln1d, "weight") and ln1d.weight is not None:
        ch_ln.weight.data = ln1d.weight.data.view(1, channels, 1, 1)
    if hasattr(ln1d, "bias") and ln1d.bias is not None:
        ch_ln.bias.data = ln1d.bias.data.view(1, channels, 1, 1)

    return ch_ln


def _is_causal_conv(conv_module: nn.Module) -> bool:
    """Check if conv module is a CausalConv1d."""
    # CausalConv1d has a 'conv' attribute and 'padding' attribute
    # Also check class name for compatibility
    cls_name = conv_module.__class__.__name__
    if "Causal" in cls_name:
        return True
    # Check if it's a wrapper with inner conv
    if hasattr(conv_module, "conv") and hasattr(conv_module, "padding"):
        # CausalConv1d uses padding=0 in inner conv and stores padding * 2
        return conv_module.conv.padding[0] == 0
    return False


def convert_cab_to_reshape_free(
    cab: nn.Module,
    axis: str,
    causal: Optional[bool] = None,
) -> ReshapeFreeCAB:
    """
    Convert Channel_Attention_Block to ReshapeFreeCAB.

    Args:
        cab: Original CAB module
        axis: Processing axis ('time' or 'freq')
        causal: If provided, use this value; otherwise auto-detect from dwconv

    Returns:
        ReshapeFreeCAB with transferred weights
    """
    # Determine parameters from original CAB
    channels = cab.norm.weight.shape[0] if hasattr(cab.norm, "weight") else 64

    # Get kernel size from dwconv
    if hasattr(cab.dwconv, "conv"):
        # CausalConv1d or Stateful version
        kernel_size = cab.dwconv.conv.kernel_size[0]
    elif hasattr(cab.dwconv, "kernel_size"):
        kernel_size = cab.dwconv.kernel_size[0]
    else:
        kernel_size = 3

    # Auto-detect causal mode if not specified
    if causal is None:
        causal = _is_causal_conv(cab.dwconv) if axis == "time" else False

    rf_cab = ReshapeFreeCAB(channels=channels, kernel_size=kernel_size, axis=axis, causal=causal)

    # Transfer weights
    # 1. LayerNorm (use ChannelLayerNorm2d to match original LayerNorm1d channel-dim normalization)
    if hasattr(cab, "norm"):
        rf_cab.norm = _convert_layernorm1d_to_channel(cab.norm)

    # 2. pwconv1: Conv1d(C, 2C, 1) → Conv2d(C, 2C, 1)
    if hasattr(cab, "pwconv1"):
        rf_cab.pwconv1.weight.data = cab.pwconv1.weight.data.unsqueeze(-1)
        if cab.pwconv1.bias is not None:
            rf_cab.pwconv1.bias.data = cab.pwconv1.bias.data.clone()

    # 3. dwconv: Conv1d → Conv2d (transfer weights only, keep existing structure)
    if hasattr(cab, "dwconv"):
        dwconv_src = cab.dwconv.conv if hasattr(cab.dwconv, "conv") else cab.dwconv
        # Transfer weights: [C, C/g, K] → [C, C/g, K, 1] (time) or [C, C/g, 1, K] (freq)
        if axis == "time":
            rf_cab.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            rf_cab.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            rf_cab.dwconv.bias.data = dwconv_src.bias.data.clone()

    # 4. sca: Sequential(Pool/CausalConv, Conv1d) → Conv2d only (pointwise part)
    if hasattr(cab, "sca") and isinstance(cab.sca, nn.Sequential) and len(cab.sca) > 1:
        # The pointwise Conv1d is the last element
        sca_conv = cab.sca[-1]
        if isinstance(sca_conv, nn.Conv1d):
            rf_cab.sca_conv.weight.data = sca_conv.weight.data.unsqueeze(-1)
            if sca_conv.bias is not None:
                rf_cab.sca_conv.bias.data = sca_conv.bias.data.clone()

    # 5. pwconv2
    if hasattr(cab, "pwconv2"):
        rf_cab.pwconv2.weight.data = cab.pwconv2.weight.data.unsqueeze(-1)
        if cab.pwconv2.bias is not None:
            rf_cab.pwconv2.bias.data = cab.pwconv2.bias.data.clone()

    # 6. beta: [1, C, 1] → [1, C, 1, 1]
    if hasattr(cab, "beta"):
        rf_cab.beta.data = cab.beta.data.unsqueeze(-1)

    return rf_cab


def convert_gpkffn_to_reshape_free(
    gpkffn: nn.Module,
    axis: str,
    causal: Optional[bool] = None,
) -> ReshapeFreeGPKFFN:
    """
    Convert Group_Prime_Kernel_FFN to ReshapeFreeGPKFFN.

    Args:
        gpkffn: Original GPKFFN module
        axis: Processing axis ('time' or 'freq')
        causal: If provided, use this value; otherwise auto-detect

    Returns:
        ReshapeFreeGPKFFN with transferred weights
    """
    channels = gpkffn.in_channel
    kernel_list = gpkffn.kernel_list

    # Auto-detect causal mode if not specified
    if causal is None and axis == "time":
        # Check first kernel's conv for causal
        first_k = kernel_list[0]
        conv_src = getattr(gpkffn, f"conv_{first_k}", None)
        causal = _is_causal_conv(conv_src) if conv_src else False
    elif causal is None:
        causal = False

    rf_gpkffn = ReshapeFreeGPKFFN(
        channels=channels,
        kernel_list=kernel_list,
        axis=axis,
        causal=causal,
    )

    # Transfer weights
    # 1. LayerNorm (use ChannelLayerNorm2d to match original LayerNorm1d channel-dim normalization)
    if hasattr(gpkffn, "norm"):
        rf_gpkffn.norm = _convert_layernorm1d_to_channel(gpkffn.norm)

    # 2. proj_first: Conv1d(C, C*4, 1) → Conv2d
    if hasattr(gpkffn, "proj_first"):
        proj_conv = gpkffn.proj_first[0] if isinstance(gpkffn.proj_first, nn.Sequential) else gpkffn.proj_first
        rf_gpkffn.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            rf_gpkffn.proj_first.bias.data = proj_conv.bias.data.clone()

    # 3. proj_last
    if hasattr(gpkffn, "proj_last"):
        proj_conv = gpkffn.proj_last[0] if isinstance(gpkffn.proj_last, nn.Sequential) else gpkffn.proj_last
        rf_gpkffn.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            rf_gpkffn.proj_last.bias.data = proj_conv.bias.data.clone()

    # 4. scale: [1, C, 1] → [1, C, 1, 1]
    if hasattr(gpkffn, "scale"):
        rf_gpkffn.scale.data = gpkffn.scale.data.unsqueeze(-1)

    # 5. Per-kernel convolutions
    for k in kernel_list:
        # attn path: Sequential(Conv1d depthwise, Conv1d pointwise)
        attn_src = getattr(gpkffn, f"attn_{k}")
        attn_dst = getattr(rf_gpkffn, f"attn_{k}")

        # Handle both regular and stateful versions
        if isinstance(attn_src, nn.Sequential):
            for i, (src_layer, dst_layer) in enumerate(zip(attn_src, attn_dst)):
                src_conv = src_layer.conv if hasattr(src_layer, "conv") else src_layer
                # Get destination conv (may be wrapped in CausalConv2dTime)
                dst_conv = dst_layer.conv if hasattr(dst_layer, "conv") else dst_layer
                if isinstance(src_conv, nn.Conv1d):
                    if axis == "time":
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(-1)
                    else:
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(2)
                    if src_conv.bias is not None:
                        dst_conv.bias.data = src_conv.bias.data.clone()

        # conv path: Conv1d depthwise
        conv_src = getattr(gpkffn, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(rf_gpkffn, f"conv_{k}")
        # Get destination conv (may be wrapped in CausalConv2dTime)
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if isinstance(conv_src_actual, nn.Conv1d):
            if axis == "time":
                conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
            else:
                conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
            if conv_src_actual.bias is not None:
                conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()

    return rf_gpkffn


def convert_ts_block_to_reshape_free(
    ts_block: nn.Module,
) -> ReshapeFreeTSBlock:
    """
    Convert TS_BLOCK to ReshapeFreeTSBlock.

    This is the main conversion function that transforms a complete TS_BLOCK
    to its reshape-free equivalent while preserving all weights.

    Args:
        ts_block: Original TS_BLOCK module

    Returns:
        ReshapeFreeTSBlock with transferred weights
    """
    dense_channel = ts_block.dense_channel

    # Determine configuration from original block
    time_block_num = len(ts_block.time_stage)
    freq_block_num = len(ts_block.freq_stage)

    # Get kernel configurations
    first_time_block = ts_block.time_stage[0]
    cab = first_time_block[0]
    gpkffn = first_time_block[1]

    # Extract kernel size from CAB
    if hasattr(cab.dwconv, "conv"):
        time_dw_kernel_size = cab.dwconv.conv.kernel_size[0]
    elif hasattr(cab.dwconv, "kernel_size"):
        time_dw_kernel_size = cab.dwconv.kernel_size[0]
    else:
        time_dw_kernel_size = 3

    time_block_kernel = gpkffn.kernel_list

    # Get freq kernel config
    first_freq_block = ts_block.freq_stage[0]
    freq_gpkffn = first_freq_block[1]
    freq_block_kernel = freq_gpkffn.kernel_list

    # Detect causal mode from time stage CAB's dwconv
    causal = _is_causal_conv(cab.dwconv)
    logger.info(f"Detected causal mode: {causal}")

    # Create reshape-free version
    rf_ts_block = ReshapeFreeTSBlock(
        dense_channel=dense_channel,
        time_block_num=time_block_num,
        freq_block_num=freq_block_num,
        time_dw_kernel_size=time_dw_kernel_size,
        time_block_kernel=time_block_kernel,
        freq_block_kernel=freq_block_kernel,
        causal=causal,
    )

    # Transfer time_stage weights
    for i, block in enumerate(ts_block.time_stage):
        cab_src = block[0]
        gpkffn_src = block[1]

        rf_ts_block.time_stage[i][0] = convert_cab_to_reshape_free(cab_src, axis="time", causal=causal)
        rf_ts_block.time_stage[i][1] = convert_gpkffn_to_reshape_free(gpkffn_src, axis="time", causal=causal)

    # Transfer freq_stage weights
    for i, block in enumerate(ts_block.freq_stage):
        cab_src = block[0]
        gpkffn_src = block[1]

        rf_ts_block.freq_stage[i][0] = convert_cab_to_reshape_free(cab_src, axis="freq")
        rf_ts_block.freq_stage[i][1] = convert_gpkffn_to_reshape_free(gpkffn_src, axis="freq")

    # Transfer beta parameters: [1, C, 1] → [1, C, 1, 1]
    rf_ts_block.beta_t.data = ts_block.beta_t.data.unsqueeze(-1)
    rf_ts_block.beta_f.data = ts_block.beta_f.data.unsqueeze(-1)

    logger.info(
        f"Converted TS_BLOCK: {time_block_num} time blocks, {freq_block_num} freq blocks"
    )

    return rf_ts_block


def convert_sequence_block_to_reshape_free(
    sequence_block: nn.Sequential,
) -> nn.Sequential:
    """
    Convert a sequence of TS_BLOCKs to reshape-free versions.

    Args:
        sequence_block: Sequential container of TS_BLOCKs

    Returns:
        Sequential container of ReshapeFreeTSBlocks
    """
    rf_blocks = []
    for i, ts_block in enumerate(sequence_block):
        rf_block = convert_ts_block_to_reshape_free(ts_block)
        rf_blocks.append(rf_block)
        logger.info(f"Converted TS_BLOCK {i}")

    return nn.Sequential(*rf_blocks)


def convert_backbone_to_reshape_free(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a complete Backbone model to reshape-free version.

    This replaces all TS_BLOCKs in the sequence_block with ReshapeFreeTSBlocks.
    The encoder and decoder modules are kept unchanged.

    Args:
        model: Original Backbone model
        inplace: If True, modify model in place; otherwise create a copy

    Returns:
        Model with reshape-free TS_BLOCKs
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Convert sequence_block (contains all TS_BLOCKs)
    if hasattr(model, "sequence_block"):
        model.sequence_block = convert_sequence_block_to_reshape_free(
            model.sequence_block
        )
        logger.info(f"Converted {len(model.sequence_block)} TS_BLOCKs to reshape-free")

    return model


def verify_conversion(
    original: nn.Module,
    converted: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 64, 40, 100),
    atol: float = 1e-5,
) -> Dict[str, any]:
    """
    Verify that the converted model produces equivalent outputs.

    Args:
        original: Original TS_BLOCK or model
        converted: Converted reshape-free version
        input_shape: Input tensor shape [B, C, T, F]
        atol: Absolute tolerance for comparison

    Returns:
        Dict with verification results
    """
    original.eval()
    converted.eval()

    x = torch.randn(*input_shape)

    with torch.no_grad():
        out_original = original(x)
        out_converted = converted(x)

    max_diff = (out_original - out_converted).abs().max().item()
    is_equal = torch.allclose(out_original, out_converted, atol=atol)

    return {
        "max_difference": max_diff,
        "is_equivalent": is_equal,
        "original_shape": list(out_original.shape),
        "converted_shape": list(out_converted.shape),
    }


def convert_ts_block_to_stateful_reshape_free(
    ts_block: nn.Module,
) -> "StatefulReshapeFreeTSBlock":
    """
    Convert TS_BLOCK to StatefulReshapeFreeTSBlock for streaming inference.

    This creates a stateful reshape-free version with:
    - Unified batch dimension (B=1 for all states)
    - Explicit state I/O (ONNX-exportable)
    - freq_stage stateless (no streaming state needed)

    Args:
        ts_block: Original TS_BLOCK module

    Returns:
        StatefulReshapeFreeTSBlock with transferred weights
    """
    from src.models.streaming.layers.reshape_free_stateful import (
        StatefulReshapeFreeTSBlock,
    )

    dense_channel = ts_block.dense_channel

    # Determine configuration from original block
    time_block_num = len(ts_block.time_stage)
    freq_block_num = len(ts_block.freq_stage)

    # Get kernel configurations
    first_time_block = ts_block.time_stage[0]
    cab = first_time_block[0]
    gpkffn = first_time_block[1]

    # Extract kernel size from CAB dwconv
    if hasattr(cab.dwconv, "conv"):
        time_dw_kernel_size = cab.dwconv.conv.kernel_size[0]
    elif hasattr(cab.dwconv, "kernel_size"):
        time_dw_kernel_size = cab.dwconv.kernel_size[0]
    else:
        time_dw_kernel_size = 3

    # Extract SCA kernel size from CAB sca
    sca_kernel_size = 11  # default
    if hasattr(cab, "sca") and isinstance(cab.sca, nn.Sequential):
        sca_first = cab.sca[0]
        # CausalConv1d wrapper: .conv attribute
        if hasattr(sca_first, "conv"):
            sca_kernel_size = sca_first.conv.kernel_size[0]
        elif hasattr(sca_first, "kernel_size"):
            sca_kernel_size = sca_first.kernel_size[0] if isinstance(sca_first.kernel_size, tuple) else sca_first.kernel_size

    time_block_kernel = gpkffn.kernel_list

    # Get freq kernel config
    first_freq_block = ts_block.freq_stage[0]
    freq_gpkffn = first_freq_block[1]
    freq_block_kernel = freq_gpkffn.kernel_list

    # Create stateful reshape-free version
    stateful_ts_block = StatefulReshapeFreeTSBlock(
        dense_channel=dense_channel,
        time_block_num=time_block_num,
        freq_block_num=freq_block_num,
        time_dw_kernel_size=time_dw_kernel_size,
        time_block_kernel=time_block_kernel,
        freq_block_kernel=freq_block_kernel,
        sca_kernel_size=sca_kernel_size,
    )

    # Transfer time_stage weights
    for i, block in enumerate(ts_block.time_stage):
        cab_src = block[0]
        gpkffn_src = block[1]

        _transfer_cab_weights_to_stateful(
            cab_src, stateful_ts_block.time_cabs[i], axis="time"
        )
        _transfer_gpkffn_weights_to_stateful(
            gpkffn_src, stateful_ts_block.time_gpkffns[i], axis="time"
        )

    # Transfer freq_stage weights (non-stateful)
    for i, block in enumerate(ts_block.freq_stage):
        cab_src = block[0]
        gpkffn_src = block[1]

        # freq_stage uses non-stateful ReshapeFreeCAB/GPKFFN
        freq_module = stateful_ts_block.freq_stage[i]
        _transfer_cab_weights_to_reshape_free(cab_src, freq_module[0], axis="freq")
        _transfer_gpkffn_weights_to_reshape_free(gpkffn_src, freq_module[1], axis="freq")

    # Transfer beta parameters: [1, C, 1] → [1, C, 1, 1]
    stateful_ts_block.beta_t.data = ts_block.beta_t.data.unsqueeze(-1)
    stateful_ts_block.beta_f.data = ts_block.beta_f.data.unsqueeze(-1)

    logger.info(
        f"Converted TS_BLOCK to Stateful Reshape-Free: "
        f"{time_block_num} time blocks, {freq_block_num} freq blocks"
    )

    return stateful_ts_block


def _transfer_cab_weights_to_stateful(
    cab_src: nn.Module,
    cab_dst: "StatefulReshapeFreeCAB",
    axis: str,
) -> None:
    """Transfer weights from original CAB to StatefulReshapeFreeCAB."""
    # 1. LayerNorm → ChannelLayerNorm2d (normalizes over C, matching original LayerNorm1d)
    if hasattr(cab_src, "norm"):
        cab_dst.norm = _convert_layernorm1d_to_channel(cab_src.norm)

    # 2. pwconv1: Conv1d → Conv2d
    if hasattr(cab_src, "pwconv1"):
        cab_dst.pwconv1.weight.data = cab_src.pwconv1.weight.data.unsqueeze(-1)
        if cab_src.pwconv1.bias is not None:
            cab_dst.pwconv1.bias.data = cab_src.pwconv1.bias.data.clone()

    # 3. dwconv: Transfer weights to StatefulReshapeFreeConv2d
    if hasattr(cab_src, "dwconv"):
        dwconv_src = cab_src.dwconv.conv if hasattr(cab_src.dwconv, "conv") else cab_src.dwconv
        dwconv_dst = cab_dst.dwconv.conv if hasattr(cab_dst.dwconv, "conv") else cab_dst.dwconv
        if axis == "time":
            dwconv_dst.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            dwconv_dst.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            dwconv_dst.bias.data = dwconv_src.bias.data.clone()

    # 4. SCA: depthwise conv + pointwise conv
    if hasattr(cab_src, "sca") and isinstance(cab_src.sca, nn.Sequential):
        sca = cab_src.sca
        # Causal SCA: Sequential(CausalConv1d(depthwise), Conv1d(pointwise))
        sca_dw_src = sca[0]
        sca_pw_src = sca[1] if len(sca) > 1 else None

        # Transfer depthwise conv weights to sca_dwconv
        if hasattr(cab_dst, "sca_dwconv") and cab_dst.sca_dwconv is not None:
            dw_src_conv = sca_dw_src.conv if hasattr(sca_dw_src, "conv") else sca_dw_src
            dw_dst_conv = cab_dst.sca_dwconv.conv
            if isinstance(dw_src_conv, nn.Conv1d):
                # [C, 1, K] → [C, 1, K, 1] (time axis)
                dw_dst_conv.weight.data = dw_src_conv.weight.data.unsqueeze(-1)
                if dw_src_conv.bias is not None:
                    dw_dst_conv.bias.data = dw_src_conv.bias.data.clone()

        # Transfer pointwise conv weights to sca_conv
        if sca_pw_src is not None and isinstance(sca_pw_src, nn.Conv1d):
            cab_dst.sca_conv.weight.data = sca_pw_src.weight.data.unsqueeze(-1)
            if sca_pw_src.bias is not None:
                cab_dst.sca_conv.bias.data = sca_pw_src.bias.data.clone()

    # 5. pwconv2
    if hasattr(cab_src, "pwconv2"):
        cab_dst.pwconv2.weight.data = cab_src.pwconv2.weight.data.unsqueeze(-1)
        if cab_src.pwconv2.bias is not None:
            cab_dst.pwconv2.bias.data = cab_src.pwconv2.bias.data.clone()

    # 6. beta: [1, C, 1] → [1, C, 1, 1]
    if hasattr(cab_src, "beta"):
        cab_dst.beta.data = cab_src.beta.data.unsqueeze(-1)


def _transfer_gpkffn_weights_to_stateful(
    gpkffn_src: nn.Module,
    gpkffn_dst: "StatefulReshapeFreeGPKFFN",
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to StatefulReshapeFreeGPKFFN."""
    # 1. LayerNorm → ChannelLayerNorm2d (normalizes over C, matching original LayerNorm1d)
    if hasattr(gpkffn_src, "norm"):
        gpkffn_dst.norm = _convert_layernorm1d_to_channel(gpkffn_src.norm)

    # 2. proj_first
    if hasattr(gpkffn_src, "proj_first"):
        proj_conv = (
            gpkffn_src.proj_first[0]
            if isinstance(gpkffn_src.proj_first, nn.Sequential)
            else gpkffn_src.proj_first
        )
        gpkffn_dst.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_first.bias.data = proj_conv.bias.data.clone()

    # 3. proj_last
    if hasattr(gpkffn_src, "proj_last"):
        proj_conv = (
            gpkffn_src.proj_last[0]
            if isinstance(gpkffn_src.proj_last, nn.Sequential)
            else gpkffn_src.proj_last
        )
        gpkffn_dst.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_last.bias.data = proj_conv.bias.data.clone()

    # 4. scale: [1, C, 1] → [1, C, 1, 1]
    if hasattr(gpkffn_src, "scale"):
        gpkffn_dst.scale.data = gpkffn_src.scale.data.unsqueeze(-1)

    # 5. Per-kernel convolutions
    kernel_list = gpkffn_src.kernel_list
    for k in kernel_list:
        # attn path: Sequential(Conv1d depthwise, Conv1d pointwise)
        attn_src = getattr(gpkffn_src, f"attn_{k}")
        attn_dw_dst = getattr(gpkffn_dst, f"attn_dw_{k}")
        attn_pw_dst = getattr(gpkffn_dst, f"attn_pw_{k}")

        if isinstance(attn_src, nn.Sequential):
            # First layer: depthwise conv
            src_dw = attn_src[0].conv if hasattr(attn_src[0], "conv") else attn_src[0]
            dst_dw = attn_dw_dst.conv if hasattr(attn_dw_dst, "conv") else attn_dw_dst
            if axis == "time":
                dst_dw.weight.data = src_dw.weight.data.unsqueeze(-1)
            else:
                dst_dw.weight.data = src_dw.weight.data.unsqueeze(2)
            if src_dw.bias is not None:
                dst_dw.bias.data = src_dw.bias.data.clone()

            # Second layer: pointwise conv
            src_pw = attn_src[1]
            attn_pw_dst.weight.data = src_pw.weight.data.unsqueeze(-1)
            if src_pw.bias is not None:
                attn_pw_dst.bias.data = src_pw.bias.data.clone()

        # conv path
        conv_src = getattr(gpkffn_src, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(gpkffn_dst, f"conv_{k}")
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if axis == "time":
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
        else:
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
        if conv_src_actual.bias is not None:
            conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()


def _transfer_cab_weights_to_reshape_free(
    cab_src: nn.Module,
    cab_dst: "ReshapeFreeCAB",
    axis: str,
) -> None:
    """Transfer weights from original CAB to ReshapeFreeCAB (non-stateful)."""
    # 1. LayerNorm (use ChannelLayerNorm2d to match original LayerNorm1d channel-dim normalization)
    if hasattr(cab_src, "norm"):
        cab_dst.norm = _convert_layernorm1d_to_channel(cab_src.norm)

    # 2. pwconv1
    if hasattr(cab_src, "pwconv1"):
        cab_dst.pwconv1.weight.data = cab_src.pwconv1.weight.data.unsqueeze(-1)
        if cab_src.pwconv1.bias is not None:
            cab_dst.pwconv1.bias.data = cab_src.pwconv1.bias.data.clone()

    # 3. dwconv
    if hasattr(cab_src, "dwconv"):
        dwconv_src = cab_src.dwconv.conv if hasattr(cab_src.dwconv, "conv") else cab_src.dwconv
        if axis == "time":
            cab_dst.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            cab_dst.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            cab_dst.dwconv.bias.data = dwconv_src.bias.data.clone()

    # 4. sca_conv: from SCA Sequential
    if hasattr(cab_src, "sca") and isinstance(cab_src.sca, nn.Sequential) and len(cab_src.sca) > 1:
        # The pointwise Conv1d is the last element in the Sequential
        sca_conv = cab_src.sca[-1]
        if isinstance(sca_conv, nn.Conv1d):
            cab_dst.sca_conv.weight.data = sca_conv.weight.data.unsqueeze(-1)
            if sca_conv.bias is not None:
                cab_dst.sca_conv.bias.data = sca_conv.bias.data.clone()

    # 5. pwconv2
    if hasattr(cab_src, "pwconv2"):
        cab_dst.pwconv2.weight.data = cab_src.pwconv2.weight.data.unsqueeze(-1)
        if cab_src.pwconv2.bias is not None:
            cab_dst.pwconv2.bias.data = cab_src.pwconv2.bias.data.clone()

    # 6. beta
    if hasattr(cab_src, "beta"):
        cab_dst.beta.data = cab_src.beta.data.unsqueeze(-1)


def _transfer_gpkffn_weights_to_reshape_free(
    gpkffn_src: nn.Module,
    gpkffn_dst: "ReshapeFreeGPKFFN",
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to ReshapeFreeGPKFFN (non-stateful)."""
    # 1. LayerNorm (use ChannelLayerNorm2d to match original LayerNorm1d channel-dim normalization)
    if hasattr(gpkffn_src, "norm"):
        gpkffn_dst.norm = _convert_layernorm1d_to_channel(gpkffn_src.norm)

    # 2. proj_first
    if hasattr(gpkffn_src, "proj_first"):
        proj_conv = (
            gpkffn_src.proj_first[0]
            if isinstance(gpkffn_src.proj_first, nn.Sequential)
            else gpkffn_src.proj_first
        )
        gpkffn_dst.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_first.bias.data = proj_conv.bias.data.clone()

    # 3. proj_last
    if hasattr(gpkffn_src, "proj_last"):
        proj_conv = (
            gpkffn_src.proj_last[0]
            if isinstance(gpkffn_src.proj_last, nn.Sequential)
            else gpkffn_src.proj_last
        )
        gpkffn_dst.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_last.bias.data = proj_conv.bias.data.clone()

    # 4. scale
    if hasattr(gpkffn_src, "scale"):
        gpkffn_dst.scale.data = gpkffn_src.scale.data.unsqueeze(-1)

    # 5. Per-kernel convolutions
    kernel_list = gpkffn_src.kernel_list
    for k in kernel_list:
        # attn path
        attn_src = getattr(gpkffn_src, f"attn_{k}")
        attn_dst = getattr(gpkffn_dst, f"attn_{k}")

        if isinstance(attn_src, nn.Sequential):
            for src_layer, dst_layer in zip(attn_src, attn_dst):
                src_conv = src_layer.conv if hasattr(src_layer, "conv") else src_layer
                dst_conv = dst_layer.conv if hasattr(dst_layer, "conv") else dst_layer
                if isinstance(src_conv, nn.Conv1d):
                    if axis == "time":
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(-1)
                    else:
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(2)
                    if src_conv.bias is not None:
                        dst_conv.bias.data = src_conv.bias.data.clone()

        # conv path
        conv_src = getattr(gpkffn_src, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(gpkffn_dst, f"conv_{k}")
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if axis == "time":
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
        else:
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
        if conv_src_actual.bias is not None:
            conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()


def convert_sequence_block_to_stateful_reshape_free(
    sequence_block: nn.Sequential,
) -> nn.ModuleList:
    """
    Convert a sequence of TS_BLOCKs to stateful reshape-free versions.

    Args:
        sequence_block: Sequential container of TS_BLOCKs

    Returns:
        ModuleList of StatefulReshapeFreeTSBlocks
    """
    rf_blocks = nn.ModuleList()
    for i, ts_block in enumerate(sequence_block):
        rf_block = convert_ts_block_to_stateful_reshape_free(ts_block)
        rf_blocks.append(rf_block)
        logger.info(f"Converted TS_BLOCK {i} to Stateful Reshape-Free")

    return rf_blocks


__all__ = [
    "convert_conv1d_to_conv2d",
    "convert_layernorm1d_to_axis",
    "convert_cab_to_reshape_free",
    "convert_gpkffn_to_reshape_free",
    "convert_ts_block_to_reshape_free",
    "convert_sequence_block_to_reshape_free",
    "convert_backbone_to_reshape_free",
    "convert_ts_block_to_stateful_reshape_free",
    "convert_sequence_block_to_stateful_reshape_free",
    "verify_conversion",
]
