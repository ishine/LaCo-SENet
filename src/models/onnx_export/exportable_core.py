"""
Exportable Neural Network Core for ONNX.

This module provides an ONNX-exportable wrapper for the LaCoSENet
neural network core (DenseEncoder + TS_BLOCK + Decoders).

The wrapper:
1. Takes mag/pha as input (STFT done on host)
2. Returns est_mask/est_pha as output (iSTFT done on host)
3. Accepts and returns all states as explicit tensors

Graph boundary (fixed design):
    Host FP32:  audio -> STFT -> mag/pha
    ONNX INT8:  mag/pha -> NNCore -> est_mask/est_pha
    Host FP32:  est_mask/est_pha -> complex -> iSTFT -> audio
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ExportableNNCore(nn.Module):
    """
    ONNX-exportable neural network core for streaming inference.

    This module wraps the DenseEncoder + TS_BLOCK + Decoders with
    explicit state I/O for ONNX export.

    Input/Output:
        Inputs:
            - mag: Magnitude spectrogram [B, F, T]
            - pha: Phase spectrogram [B, F, T]
            - *states: All state tensors (conv states)
            - state_frames: Number of frames for state update (optional, can be constant)

        Outputs:
            - est_mask: Estimated mask [B, F, T]
            - est_pha: Estimated phase [B, F, T]
            - *next_states: Updated state tensors

    Example:
        >>> from src.models.onnx_export import ExportableNNCore
        >>> core = ExportableNNCore.from_lacosenet(streaming_model)
        >>> states = core.init_states(batch_size=1)
        >>> est_mask, est_pha, *next_states = core(mag, pha, *states)
    """

    def __init__(
        self,
        dense_encoder: nn.Module,
        sequence_block: nn.Module,
        mask_decoder: nn.Module,
        phase_decoder: nn.Module,
        infer_type: str = "masking",
    ):
        """
        Initialize ExportableNNCore.

        Note: Use from_lacosenet() or from_checkpoint() for easier creation.

        Args:
            dense_encoder: DenseEncoder with functional stateful convs
            sequence_block: TS_BLOCK sequence with functional SCA
            mask_decoder: MaskDecoder with functional stateful convs
            phase_decoder: PhaseDecoder with functional stateful convs
            infer_type: "masking" or "mapping"
        """
        super().__init__()
        self.dense_encoder = dense_encoder
        self.sequence_block = sequence_block
        self.mask_decoder = mask_decoder
        self.phase_decoder = phase_decoder
        self.infer_type = infer_type

        # State management (populated by _collect_states)
        self._state_modules: List[Tuple[str, nn.Module]] = []
        self._state_count = 0

    def _collect_state_modules(self) -> None:
        """Collect all modules with init_state method."""
        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
        )

        stateful_types = (
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
            FunctionalStatefulCausalConv2d,
        )

        self._state_modules = []
        for name, module in self.named_modules():
            if isinstance(module, stateful_types):
                self._state_modules.append((name, module))

        self._state_count = len(self._state_modules)
        logger.info(f"Collected {self._state_count} stateful modules")

    @property
    def state_count(self) -> int:
        """Number of state tensors."""
        if self._state_count == 0:
            self._collect_state_modules()
        return self._state_count

    def init_states(
        self,
        batch_size: int = 1,
        freq_size: int = 129,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """
        Initialize all state tensors.

        Args:
            batch_size: Batch size
            freq_size: Frequency dimension (n_fft // 2 + 1)
            device: Device for states
            dtype: Data type for states

        Returns:
            List of initialized state tensors
        """
        if self._state_count == 0:
            self._collect_state_modules()

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
        )

        states = []
        for name, module in self._state_modules:
            if isinstance(module, FunctionalStatefulConv1d):
                state = module.init_state(batch_size, device, dtype)
            elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                state = module.init_state(batch_size, freq_size, device, dtype)
            else:
                raise TypeError(f"Unknown stateful module: {type(module)}")
            states.append(state)

        return states

    def get_state_names(self) -> List[str]:
        """Get names of all state tensors."""
        if self._state_count == 0:
            self._collect_state_modules()
        return [f"state_{i}_{name.replace('.', '_')}" for i, (name, _) in enumerate(self._state_modules)]

    def forward(
        self,
        mag: Tensor,
        pha: Tensor,
        *states: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass with explicit state I/O.

        Note: This is a simplified forward that processes the entire sequence.
        For true streaming with per-layer state management, use the step-by-step
        approach or the host wrapper.

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]
            *states: Previous state tensors
            state_frames: Number of frames for state update

        Returns:
            Tuple of (est_mask, est_pha, *next_states)
        """
        # Convert to model input format [B, 2, T, F]
        B, F, T = mag.shape
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)

        # Process through encoder
        x = self.dense_encoder(x)

        # Process through TS_BLOCK
        x = self.sequence_block(x)

        # Process through decoders
        mask = self.mask_decoder(x).squeeze(1).transpose(1, 2)  # [B, F, T]
        est_pha = self.phase_decoder(x).squeeze(1).transpose(1, 2)  # [B, F, T]

        # Apply mask if masking mode
        if self.infer_type == "masking":
            est_mask = mask
        else:
            est_mask = mask

        # For now, pass states through unchanged
        # Full state management would require layer-by-layer explicit calls
        return (est_mask, est_pha) + tuple(states)

    @classmethod
    def from_backbone(
        cls,
        model: nn.Module,
        convert_layers: bool = True,
    ) -> "ExportableNNCore":
        """
        Create ExportableNNCore from a Backbone model.

        Args:
            model: Backbone model (with or without stateful conversions)
            convert_layers: If True, convert to ONNX-friendly layers

        Returns:
            ExportableNNCore instance
        """
        import copy

        from src.models.onnx_export.layers.conv_transpose_wrapper import (
            convert_conv_transpose_to_wrapper,
        )

        # Deep copy to avoid modifying original
        model = copy.deepcopy(model)
        model.eval()

        if convert_layers:
            # Convert ConvTranspose2d to ONNX-friendly wrapper
            model, tconv_count = convert_conv_transpose_to_wrapper(model, inplace=True)
            logger.info(f"Converted {tconv_count} ConvTranspose2d to wrapper")

        # Extract components
        core = cls(
            dense_encoder=model.dense_encoder,
            sequence_block=model.sequence_block,
            mask_decoder=model.mask_decoder,
            phase_decoder=model.phase_decoder,
            infer_type=getattr(model, 'infer_type', 'masking'),
        )

        return core


def export_nncore_to_onnx(
    core: ExportableNNCore,
    output_path: str,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verbose: bool = True,
) -> str:
    """
    Export ExportableNNCore to ONNX format.

    Note: This exports the stateless version. For stateful export,
    use export_stateful_nncore_to_onnx (TODO).

    Args:
        core: ExportableNNCore instance
        output_path: Path for output ONNX file
        batch_size: Batch size for export
        time_frames: Number of time frames
        freq_size: Frequency dimension size
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification
        verbose: Print export info

    Returns:
        Path to exported ONNX file
    """
    core.eval()
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Create dummy inputs
    mag = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)
    pha = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)

    # Default dynamic axes for streaming flexibility
    if dynamic_axes is None:
        dynamic_axes = {
            "mag": {0: "batch", 2: "time"},
            "pha": {0: "batch", 2: "time"},
            "est_mask": {0: "batch", 2: "time"},
            "est_pha": {0: "batch", 2: "time"},
        }

    # Export
    input_names = ["mag", "pha"]
    output_names = ["est_mask", "est_pha"]

    if verbose:
        print(f"Exporting to: {output_path}")
        print(f"  Input shape: mag/pha [{batch_size}, {freq_size}, {time_frames}]")
        print(f"  Opset version: {opset_version}")

    # Create parent directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        core,
        (mag, pha),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Exported: {file_size:.2f} MB")

    return output_path


def verify_onnx_export(
    onnx_path: str,
    core: ExportableNNCore,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Verify ONNX export against PyTorch model.

    Args:
        onnx_path: Path to ONNX file
        core: Original ExportableNNCore
        batch_size: Batch size for test
        time_frames: Number of time frames
        freq_size: Frequency dimension
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with verification results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    core.eval()
    device = next(core.parameters()).device

    # Create test inputs
    mag = torch.randn(batch_size, freq_size, time_frames, device=device)
    pha = torch.randn(batch_size, freq_size, time_frames, device=device)

    # PyTorch inference
    with torch.no_grad():
        pt_outputs = core(mag, pha)
        pt_mask = pt_outputs[0].cpu().numpy()
        pt_pha = pt_outputs[1].cpu().numpy()

    # ONNX Runtime inference
    sess = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "mag": mag.cpu().numpy(),
        "pha": pha.cpu().numpy(),
    }
    ort_outputs = sess.run(None, ort_inputs)
    ort_mask = ort_outputs[0]
    ort_pha = ort_outputs[1]

    # Compare
    import numpy as np

    mask_match = np.allclose(pt_mask, ort_mask, atol=atol, rtol=rtol)
    pha_match = np.allclose(pt_pha, ort_pha, atol=atol, rtol=rtol)

    mask_max_diff = np.abs(pt_mask - ort_mask).max()
    pha_max_diff = np.abs(pt_pha - ort_pha).max()

    return {
        "mask_match": mask_match,
        "pha_match": pha_match,
        "mask_max_diff": float(mask_max_diff),
        "pha_max_diff": float(pha_max_diff),
        "all_match": mask_match and pha_match,
    }


__all__ = [
    "ExportableNNCore",
    "export_nncore_to_onnx",
    "verify_onnx_export",
]
