"""
Streaming inference modules for Backbone.

This package provides streaming-compatible wrappers and utilities for
real-time audio enhancement with Backbone models.

Package Structure:
    - layers/: Streaming-compatible layer implementations
        - StatefulConv*
    - converters/: Model transformation utilities
        - convert_to_stateful
    - dublonet.py: DuBLoNet streaming wrapper
    - utils.py: Core utilities (StateFramesContext, prepare_streaming_model)

DuBLoNet Configuration:
    - **Fully Causal** (enc/dec lookahead=0): Immediate output, no buffering delay
    - **Asymmetric Decoder** (dec lookahead>0): Buffered processing
    - **Asymmetric Encoder+Decoder** (both>0): Full lookahead buffering

Example (fully causal model - immediate output):
    >>> from src.models.streaming import DuBLoNet
    >>> streaming = DuBLoNet.from_checkpoint(
    ...     "path/to/checkpoint",
    ...     encoder_lookahead=0,
    ...     decoder_lookahead=0,  # Immediate mode
    ... )
    >>> for chunk in audio_stream:
    ...     output = streaming.process_samples(chunk)

Example (asymmetric model with encoder/decoder lookahead):
    >>> from src.models.streaming import DuBLoNet
    >>> streaming = DuBLoNet.from_checkpoint(
    ...     "path/to/checkpoint",
    ...     encoder_lookahead=7,   # >0 if encoder has asymmetric padding
    ...     decoder_lookahead=7,   # >0 if decoder has asymmetric padding
    ... )

    >>> # StateFramesContext for lookahead handling (used internally)
    >>> from src.models.streaming.utils import StateFramesContext
    >>> with StateFramesContext(64):
    ...     output = model(extended_input)
"""

# =============================================================================
# Layers (used externally by enhance.py)
# =============================================================================
from .layers import (
    StatefulCausalConv1d,
    StatefulAsymmetricConv2d,
    StatefulCausalConv2d,
)

# =============================================================================
# Converters (used externally by enhance.py)
# =============================================================================
from .converters import (
    convert_to_stateful,
    set_streaming_mode,
    reset_streaming_state,
    get_stateful_layer_count,
)

# =============================================================================
# Wrappers
# =============================================================================
from .dublonet import DuBLoNet

__all__ = [
    # === Layers ===
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
    # === Converters ===
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "get_stateful_layer_count",
    # === Wrappers ===
    "DuBLoNet",
]
