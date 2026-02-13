"""
Streaming-compatible layer implementations.

This package provides drop-in replacements for standard layers that enable
streaming inference:

- StatefulCausalConv1d/2d: State-buffering convolutions
- Reshape-Free layers: Conv2d-based layers eliminating reshape operations

For ONNX-exportable layers (FunctionalStateful*, ConvTranspose2dWrapper),
see src.models.onnx_export.layers.

Example:
    >>> from src.models.streaming.layers import (
    ...     StatefulAsymmetricConv2d,
    ...     ReshapeFreeTSBlock,
    ... )
"""

from .reshape_free import (
    AxisLayerNorm,
    CausalConv2dTime,
    ReshapeFreeCAB,
    ReshapeFreeGPKFFN,
    ReshapeFreeTSBlock,
    SimpleGate2d,
)
from .reshape_free_stateful import (
    StatefulReshapeFreeCAB,
    StatefulReshapeFreeConv2d,
    StatefulReshapeFreeGPKFFN,
    StatefulReshapeFreeTSBlock,
)
from .stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)

__all__ = [
    # Convolutions (original stateful)
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
    # Reshape-Free layers (batch_size=1 optimized)
    "AxisLayerNorm",
    "SimpleGate2d",
    "CausalConv2dTime",
    "ReshapeFreeCAB",
    "ReshapeFreeGPKFFN",
    "ReshapeFreeTSBlock",
    # Reshape-Free stateful layers
    "StatefulReshapeFreeConv2d",
    "StatefulReshapeFreeCAB",
    "StatefulReshapeFreeGPKFFN",
    "StatefulReshapeFreeTSBlock",
]
