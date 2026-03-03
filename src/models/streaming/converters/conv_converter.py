"""
Stateful Convolution Conversion Utilities.

This module provides functions to convert standard Backbone models to
stateful versions for streaming inference, and utilities to manage
streaming state.

Key Functions:
    - convert_to_stateful: Convert model's conv layers to stateful versions
    - set_streaming_mode: Enable/disable streaming mode for all layers
    - reset_streaming_state: Reset all streaming states for new utterance

For the Stateful convolution layers, see:
    src.models.streaming.layers.stateful_conv

Example:
    >>> from src.models.streaming.converters import (
    ...     convert_to_stateful,
    ...     set_streaming_mode,
    ...     reset_streaming_state
    ... )
    >>>
    >>> # Convert model
    >>> stateful_model = convert_to_stateful(model)
    >>>
    >>> # Process audio stream
    >>> set_streaming_mode(stateful_model, True)
    >>> for chunk in audio_chunks:
    ...     output = stateful_model(chunk)
    >>>
    >>> # New utterance
    >>> reset_streaming_state(stateful_model)
"""

from __future__ import annotations

import copy
import logging
import warnings
from typing import Callable, Dict, Type

import torch.nn as nn

from src.models.backbone import AsymmetricConv2d, CausalConv1d, CausalConv2d
from src.models.streaming.layers.stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)

logger = logging.getLogger(__name__)


def _replace_modules_recursive(
    parent: nn.Module,
    target_type: Type[nn.Module],
    factory_fn: Callable[[nn.Module], nn.Module],
) -> int:
    """
    Safely replace modules of a specific type recursively.

    Uses named_children() based recursion to avoid iterator instability
    issues that occur when modifying modules during named_modules() iteration.

    Args:
        parent: Parent module to search in
        target_type: Type of modules to replace
        factory_fn: Factory function that creates replacement module

    Returns:
        Number of modules replaced
    """
    replaced_count = 0

    # list() to create a copy, ensuring iterator stability
    for name, child in list(parent.named_children()):
        if isinstance(child, target_type):
            # Direct replacement using setattr
            new_module = factory_fn(child)
            setattr(parent, name, new_module)
            replaced_count += 1
        else:
            # Recursively search child modules
            replaced_count += _replace_modules_recursive(child, target_type, factory_fn)

    return replaced_count


def convert_to_stateful(
    model: nn.Module,
    verbose: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert a Backbone model to stateful version for streaming.

    Recursively replaces:
    - CausalConv1d -> StatefulCausalConv1d
    - AsymmetricConv2d -> StatefulAsymmetricConv2d
    - CausalConv2d -> StatefulCausalConv2d

    Args:
        model: Original Backbone model
        verbose: If True, print conversion statistics
        inplace: If True, modify model in place (default: False, creates deep copy)

    Returns:
        Stateful version with same weights

    Note:
        The returned model has streaming mode disabled by default.
        Call set_streaming_mode(model, True) to enable streaming.
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Replace CausalConv1d
    count_1d = _replace_modules_recursive(
        model, CausalConv1d, StatefulCausalConv1d.from_causal_conv
    )

    # Replace AsymmetricConv2d
    count_asym = _replace_modules_recursive(
        model, AsymmetricConv2d, StatefulAsymmetricConv2d.from_asymmetric_conv
    )

    # Replace CausalConv2d
    count_2d = _replace_modules_recursive(
        model, CausalConv2d, StatefulCausalConv2d.from_causal_conv2d
    )

    total_count = count_1d + count_asym + count_2d

    if verbose:
        logger.info(
            f"Converted to stateful: "
            f"{count_1d} CausalConv1d, "
            f"{count_asym} AsymmetricConv2d, "
            f"{count_2d} CausalConv2d "
            f"(total: {total_count})"
        )

    if total_count == 0:
        warnings.warn(
            "No convertible layers found in model. "
            "Make sure the model contains CausalConv1d, AsymmetricConv2d, or CausalConv2d layers.",
            UserWarning,
        )

    return model


def set_streaming_mode(model: nn.Module, streaming: bool) -> int:
    """
    Set streaming mode for all stateful layers.

    Args:
        model: Stateful model
        streaming: True to enable streaming, False to disable

    Returns:
        Number of layers with streaming mode set

    Raises:
        RuntimeError: If model is in training mode and streaming=True
    """
    if streaming and model.training:
        raise RuntimeError(
            "Cannot enable streaming mode while model is in training mode. "
            "Call model.eval() first."
        )

    streaming_layers = 0
    for module in model.modules():
        if hasattr(module, "set_streaming"):
            module.set_streaming(streaming)
            streaming_layers += 1

    if streaming_layers == 0:
        warnings.warn(
            "No stateful layers found in model. "
            "Did you forget to call convert_to_stateful()?",
            UserWarning,
        )

    return streaming_layers


def reset_streaming_state(model: nn.Module) -> int:
    """
    Reset all streaming states for new utterance.

    Call this function before processing a new audio stream to clear
    all accumulated state from previous processing.

    Args:
        model: Stateful model

    Returns:
        Number of layers reset
    """
    reset_count = 0
    for module in model.modules():
        if hasattr(module, "reset_state"):
            module.reset_state()
            reset_count += 1

    return reset_count


def get_stateful_layer_count(model: nn.Module) -> Dict[str, int]:
    """
    Count stateful layers by type.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with counts per layer type
    """
    counts = {
        "StatefulCausalConv1d": 0,
        "StatefulAsymmetricConv2d": 0,
        "StatefulCausalConv2d": 0,
        "total": 0,
    }

    for module in model.modules():
        if isinstance(module, StatefulCausalConv1d):
            counts["StatefulCausalConv1d"] += 1
            counts["total"] += 1
        elif isinstance(module, StatefulAsymmetricConv2d):
            counts["StatefulAsymmetricConv2d"] += 1
            counts["total"] += 1
        elif isinstance(module, StatefulCausalConv2d):
            counts["StatefulCausalConv2d"] += 1
            counts["total"] += 1

    return counts
