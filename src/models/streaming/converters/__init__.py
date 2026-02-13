"""
Model conversion utilities for streaming inference.

This package provides functions to transform standard models into
streaming-compatible versions:

- conv_converter: Conv -> StatefulConv
- reshape_free_converter: TS_BLOCK -> ReshapeFreeTSBlock (batch_size=1 optimized)

Example:
    >>> from src.models.streaming.converters import (
    ...     convert_to_stateful,
    ...     convert_ts_block_to_reshape_free,
    ... )
"""

from .conv_converter import (
    convert_to_stateful,
    get_stateful_layer_count,
    get_total_state_size,
    reset_streaming_state,
    set_streaming_mode,
    verify_stateful_conversion,
)
from .reshape_free_converter import (
    convert_cab_to_reshape_free,
    convert_conv1d_to_conv2d,
    convert_gpkffn_to_reshape_free,
    convert_backbone_to_reshape_free,
    convert_ts_block_to_reshape_free,
    verify_conversion as verify_reshape_free_conversion,
)

__all__ = [
    # Convolution conversion
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "get_total_state_size",
    "get_stateful_layer_count",
    "verify_stateful_conversion",
    # Reshape-Free conversion (batch_size=1 optimized)
    "convert_conv1d_to_conv2d",
    "convert_cab_to_reshape_free",
    "convert_gpkffn_to_reshape_free",
    "convert_ts_block_to_reshape_free",
    "convert_backbone_to_reshape_free",
    "verify_reshape_free_conversion",
]
