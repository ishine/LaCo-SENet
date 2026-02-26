"""
ONNX Export utilities for LaCoSENet.

This package provides tools for exporting the neural network core of
LaCoSENet to ONNX format.

Key Components:
    - ExportableNNCore: ONNX-exportable wrapper (stateless version)
    - StatefulExportableNNCore: Full stateful ONNX export with explicit state I/O
    - export_nncore_to_onnx: Export function for stateless version
    - export_stateful_nncore_to_onnx: Export function for stateful version
    - StateRegistry: Manages state tensor I/O

Design Decisions (fixed):
    - STFT/iSTFT, atan2, sqrt, complex operations: Host FP32 (not in ONNX graph)
    - DenseEncoder, TS_BLOCK, MaskDecoder: ONNX graph
"""

from .exportable_core import ExportableNNCore, export_nncore_to_onnx, verify_onnx_export
from .state_registry import StateInfo, StateRegistry
from .stateful_core import (
    StatefulExportableNNCore,
    StateIterator,
    convert_stateful_to_functional,
    export_stateful_nncore_to_onnx,
    verify_stateful_onnx_export,
)
from .stateful_core_rf import (
    StatefulReshapeFreeExportableNNCore,
    export_stateful_rf_nncore_to_onnx,
    verify_stateful_rf_onnx_export,
)
from .streaming_wrapper import (
    ONNXLaCoSENet,
    QNNConfig,
    STFTConfig,
    create_ort_session,
)

__all__ = [
    # Stateless export
    "ExportableNNCore",
    "export_nncore_to_onnx",
    "verify_onnx_export",
    # Stateful export
    "StatefulExportableNNCore",
    "StateIterator",
    "convert_stateful_to_functional",
    "export_stateful_nncore_to_onnx",
    "verify_stateful_onnx_export",
    # Stateful reshape-free export
    "StatefulReshapeFreeExportableNNCore",
    "export_stateful_rf_nncore_to_onnx",
    "verify_stateful_rf_onnx_export",
    # State registry
    "StateInfo",
    "StateRegistry",
    # Streaming wrapper
    "ONNXLaCoSENet",
    "STFTConfig",
    # QNN Execution Provider support
    "QNNConfig",
    "create_ort_session",
]
