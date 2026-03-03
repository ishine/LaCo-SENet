"""
Model conversion utilities for streaming inference.

This package provides functions to transform standard models into
streaming-compatible versions:

- conv_converter: Conv -> StatefulConv
- reshape_free_converter: TS_BLOCK -> StatefulReshapeFreeTSBlock

Example:
    >>> from src.models.streaming.converters import (
    ...     convert_to_stateful,
    ...     set_streaming_mode,
    ... )
"""

from .conv_converter import (
    convert_to_stateful,
    get_stateful_layer_count,
    reset_streaming_state,
    set_streaming_mode,
)

__all__ = [
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "get_stateful_layer_count",
]
