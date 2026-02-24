"""
CPU inference optimizations: BN folding.

Provides BN folding optimization for CPU streaming inference:

**BN Folding**: Fuses Conv2d+BatchNorm2d (and ConvTranspose2d+BatchNorm2d)
pairs into single convolution layers, eliminating BN computation overhead.

Usage:
    >>> from src.models.streaming.cpu_optimizations import optimize_for_cpu
    >>> model, stats = optimize_for_cpu(model)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d + BatchNorm2d into a single Conv2d.

    Absorbs BN parameters (gamma, beta, running_mean, running_var) into the
    convolution weights and bias so that the fused layer produces identical
    output without the separate BN forward pass.

    Args:
        conv: Conv2d layer (must immediately precede bn)
        bn: BatchNorm2d layer

    Returns:
        New Conv2d with fused weights/bias.
    """
    fused = copy.deepcopy(conv)
    fused = fused.to(conv.weight.device)

    # BN parameters
    gamma = bn.weight  # [C]
    beta = bn.bias  # [C]
    mean = bn.running_mean  # [C]
    var = bn.running_var  # [C]
    eps = bn.eps

    # scale = gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(var + eps)  # [C]

    # Conv2d weight shape: [out_channels, in_channels/groups, kH, kW]
    # Scale along output channel dimension (dim=0)
    fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

    if conv.bias is not None:
        fused.bias.data = (conv.bias.data - mean) * scale + beta
    else:
        fused.bias = nn.Parameter((- mean) * scale + beta)

    return fused


def fuse_convtranspose_bn(
    conv_t: nn.ConvTranspose2d, bn: nn.BatchNorm2d
) -> nn.ConvTranspose2d:
    """Fuse ConvTranspose2d + BatchNorm2d into a single ConvTranspose2d.

    ConvTranspose2d weight shape is [in_channels, out_channels/groups, kH, kW],
    so the output channel dimension is dim=1.

    Args:
        conv_t: ConvTranspose2d layer
        bn: BatchNorm2d layer

    Returns:
        New ConvTranspose2d with fused weights/bias.
    """
    fused = copy.deepcopy(conv_t)
    fused = fused.to(conv_t.weight.device)

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    # ConvTranspose2d weight: [in_channels, out_channels/groups, kH, kW]
    # Output channels are on dim=1
    fused.weight.data = conv_t.weight.data * scale.view(1, -1, 1, 1)

    if conv_t.bias is not None:
        fused.bias.data = (conv_t.bias.data - mean) * scale + beta
    else:
        fused.bias = nn.Parameter((- mean) * scale + beta)

    return fused


def fold_batchnorm(model: nn.Module) -> Tuple[nn.Module, int]:
    """Recursively fuse all Conv+BN pairs inside nn.Sequential containers.

    Traverses the module tree and, for every nn.Sequential, looks for
    consecutive (Conv2d, BatchNorm2d) or (ConvTranspose2d, BatchNorm2d) pairs.
    Each pair is replaced by a single fused convolution, and the BN layer is
    replaced with nn.Identity().

    Args:
        model: Model to optimize (modified in-place).

    Returns:
        Tuple of (model, fused_count).
    """
    fused_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue

        children = list(module.children())
        i = 0
        while i < len(children) - 1:
            curr, nxt = children[i], children[i + 1]

            if isinstance(curr, nn.Conv2d) and isinstance(nxt, nn.BatchNorm2d):
                fused = fuse_conv_bn(curr, nxt)
                module[i] = fused
                module[i + 1] = nn.Identity()
                fused_count += 1
                children = list(module.children())
            elif isinstance(curr, nn.ConvTranspose2d) and isinstance(nxt, nn.BatchNorm2d):
                fused = fuse_convtranspose_bn(curr, nxt)
                module[i] = fused
                module[i + 1] = nn.Identity()
                fused_count += 1
                children = list(module.children())
            else:
                i += 1

    return model, fused_count


def optimize_for_cpu(
    model: nn.Module, verbose: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Apply CPU optimizations: BN folding.

    Args:
        model: Backbone model (should be in eval mode).
        verbose: Print optimization summary.

    Returns:
        Tuple of (optimized_model, stats_dict).
    """
    stats: Dict[str, Any] = {}

    model, fused_count = fold_batchnorm(model)
    stats["bn_fused"] = fused_count

    if verbose:
        print(f"  CPU Optimizations Applied:")
        print(f"    - BN Folding: {fused_count} pairs fused")

    return model, stats
