"""
ONNX-exportable layer implementations.

This package provides layers specifically designed for ONNX export:

- FunctionalStateful*: Explicit state I/O convolutions for ONNX graphs
- ConvTranspose2dWrapper: ONNX/INT8-friendly ConvTranspose2d replacement
"""

from .conv_transpose_wrapper import (
    ConvTranspose2dWrapper,
    ZeroInsertUpsample2d,
    convert_conv_transpose_to_wrapper,
    tconv_weight_to_conv_weight,
)
from .functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    convert_to_functional,
)

__all__ = [
    # Convolutions (functional/ONNX-exportable)
    "FunctionalStatefulConv1d",
    "FunctionalStatefulConv2d",
    "FunctionalStatefulCausalConv2d",
    "convert_to_functional",
    # ConvTranspose2d replacement
    "ConvTranspose2dWrapper",
    "ZeroInsertUpsample2d",
    "tconv_weight_to_conv_weight",
    "convert_conv_transpose_to_wrapper",
]
