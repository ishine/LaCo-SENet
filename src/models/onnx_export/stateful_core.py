"""
Stateful Exportable Neural Network Core for ONNX.

This module provides a fully stateful ONNX-exportable wrapper for Backbone
that explicitly passes state tensors through all layers.

Key features:
1. All stateful layers (Conv) use explicit state I/O
2. State tensors are graph inputs/outputs for ONNX export
3. No internal buffering - pure functional computation

Graph boundary (fixed design):
    Host FP32:  audio -> STFT -> mag/pha
    ONNX INT8:  mag/pha + prev_states -> NNCore -> est_mask/est_pha + next_states
    Host FP32:  est_mask/est_pha -> complex -> iSTFT -> audio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class StateSpec:
    """Specification for a state tensor."""
    name: str
    module_path: str
    shape_template: Tuple[str, ...]  # e.g., ("B", "C", "pad") or ("B*F", "C", "1")
    channels: int
    extra_dims: Dict[str, int]  # e.g., {"pad": 4, "freq_padded": 131}


class StateIterator:
    """
    Iterator for managing state tensors during forward pass.

    This class provides a clean interface for consuming previous states
    and collecting next states during the forward pass.
    """

    def __init__(self, prev_states: List[Tensor]):
        self._prev_states = list(prev_states)
        self._next_states: List[Tensor] = []
        self._idx = 0

    def get_and_update(self, next_state: Tensor) -> Tensor:
        """Get previous state and store next state."""
        if self._idx >= len(self._prev_states):
            raise IndexError(f"State index {self._idx} out of range")
        prev = self._prev_states[self._idx]
        self._next_states.append(next_state)
        self._idx += 1
        return prev

    @property
    def next_states(self) -> List[Tensor]:
        return self._next_states

    @property
    def consumed_count(self) -> int:
        return self._idx


class StatefulExportableNNCore(nn.Module):
    """
    Fully stateful ONNX-exportable neural network core.

    Unlike ExportableNNCore (which passes states unchanged), this module
    explicitly routes state tensors through each layer during forward.

    Architecture mirrors Backbone:
        mag/pha -> DenseEncoder -> TS_BLOCK x N -> MaskDecoder/PhaseDecoder

    Each stateful layer (FunctionalStatefulConv)
    receives its state as input and returns updated state as output.

    Input/Output:
        Inputs:
            - mag: Magnitude spectrogram [B, F, T]
            - pha: Phase spectrogram [B, F, T]
            - *prev_states: All previous state tensors

        Outputs:
            - est_mask: Estimated mask [B, F, T]
            - est_pha: Estimated phase [B, F, T]
            - *next_states: All updated state tensors
    """

    def __init__(
        self,
        dense_encoder: nn.Module,
        sequence_block: nn.Module,
        mask_decoder: nn.Module,
        phase_decoder: nn.Module,
        infer_type: str = "masking",
        n_fft: int = 400,
        phase_output_mode: str = "atan2",
    ):
        """
        Initialize StatefulExportableNNCore.

        Note: Use from_backbone() for easier creation.

        Args:
            dense_encoder: DenseEncoder (converted to functional layers)
            sequence_block: TS_BLOCK sequence (converted to functional layers)
            mask_decoder: MaskDecoder (converted to functional layers)
            phase_decoder: PhaseDecoder (converted to functional layers)
            infer_type: "masking" or "mapping"
            n_fft: FFT size for frequency dimension calculation
            phase_output_mode: Phase output mode:
                - "atan2": Output atan2(imag, real) phase (default, for backwards compat)
                - "complex": Output (real, imag) separately for host-side atan2
                  This is recommended for INT8 quantization to preserve phase precision.
        """
        super().__init__()
        self.dense_encoder = dense_encoder
        self.sequence_block = sequence_block
        self.mask_decoder = mask_decoder
        self.phase_decoder = phase_decoder
        self.infer_type = infer_type
        self.n_fft = n_fft
        self.phase_output_mode = phase_output_mode

        # Optional limit for state updates when processing extended (lookahead) inputs.
        # When set, functional stateful conv layers update next_state using only the
        # first `state_frames_for_update` frames, while still producing outputs for
        # the full input length. This mirrors StateFramesContext behavior used by
        # the PyTorch streaming wrapper.
        self.state_frames_for_update: Optional[int] = None

        # Collect all functional stateful modules in forward order
        self._functional_modules: List[Tuple[str, nn.Module]] = []
        self._state_specs: List[StateSpec] = []
        self._collect_functional_modules()

    def _collect_functional_modules(self) -> None:
        """Collect all functional stateful modules in DFS order."""
        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
        )

        functional_types = (
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
            FunctionalStatefulCausalConv2d,
        )

        self._functional_modules = []
        collected_names = set()

        for name, module in self.named_modules():
            if isinstance(module, functional_types):
                # Skip if this module is a child of an already collected module
                # (e.g., skip sca.squeeze if sca is already collected)
                is_nested = any(name.startswith(parent + '.') for parent in collected_names)
                if not is_nested:
                    self._functional_modules.append((name, module))
                    collected_names.add(name)

        logger.info(f"Collected {len(self._functional_modules)} functional modules")
        for name, mod in self._functional_modules:
            logger.debug(f"  {name}: {type(mod).__name__}")

    @property
    def num_states(self) -> int:
        """Number of state tensors."""
        return len(self._functional_modules)

    def get_state_names(self) -> List[str]:
        """Get names of all state tensors."""
        return [f"state_{i}_{name.replace('.', '_')}"
                for i, (name, _) in enumerate(self._functional_modules)]

    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """
        Set a fixed number of frames to use for state updates.

        Args:
            state_frames: Number of frames (from the start of the input) allowed to
                update conv states. If None, all frames update state (default).
        """
        if state_frames is None:
            self.state_frames_for_update = None
            return
        self.state_frames_for_update = int(state_frames)

    def init_states(
        self,
        batch_size: int = 1,
        freq_size: int = 129,
        time_frames: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """
        Initialize all state tensors.

        Args:
            batch_size: Batch size
            freq_size: Frequency dimension (n_fft // 2 + 1)
            time_frames: Number of time frames (for TS_BLOCK freq_stage)
            device: Device for states
            dtype: Data type for states

        Returns:
            List of initialized state tensors

        Note:
            In TS_BLOCK, batch dimension changes due to reshape:
            - time_stage: B*F_enc (F_enc is encoded frequency dimension)
            - freq_stage: B*T (T is time frames)

            Encoder/Decoder 2D convs use different freq dimensions:
            - dense_encoder: uses original freq_size
            - sequence_block, mask_decoder, phase_decoder: uses freq_enc
        """
        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv1d,
            FunctionalStatefulConv2d,
        )

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        # Calculate encoded frequency dimension
        # DenseEncoder has stride=(1,2) in dense_conv_2, so F_enc ≈ (freq_size + 2) // 2
        # More precisely: output = floor((input + 2*padding - kernel) / stride) + 1
        # For Conv2d(C, C, (1, 3), (1, 2)): F_enc = (freq_size - 3) // 2 + 1 = (freq_size - 1) // 2
        freq_enc = (freq_size - 1) // 2

        states = []
        for name, module in self._functional_modules:
            # Determine effective batch size based on module location
            if "time_stage" in name:
                # time_stage processes [B*F_enc, C, T], so batch = B * F_enc
                effective_batch = batch_size * freq_enc
            elif "freq_stage" in name:
                # freq_stage processes [B*T, C, F], so batch = B * T
                effective_batch = batch_size * time_frames
            else:
                # Encoder/Decoder: normal batch size
                effective_batch = batch_size

            # Determine frequency size for 2D conv states
            # DenseEncoder uses original freq_size, others use freq_enc
            if "dense_encoder" in name:
                conv2d_freq = freq_size
            else:
                # sequence_block, mask_decoder, phase_decoder use encoded freq
                conv2d_freq = freq_enc

            if isinstance(module, FunctionalStatefulConv1d):
                # 1D conv: [B, C, padding_size]
                state = module.init_state(effective_batch, device, dtype)
            elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                # 2D conv: [B, C, time_padding, freq_padded]
                state = module.init_state(batch_size, conv2d_freq, device, dtype)
            else:
                raise TypeError(f"Unknown module type: {type(module)}")
            states.append(state)

        return states

    def forward(
        self,
        mag: Tensor,
        pha: Tensor,
        *prev_states: Tensor,
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass with explicit state I/O.

        This method routes state tensors through all functional layers
        in the correct order.

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]
            *prev_states: Previous state tensors (must match num_states)

        Returns:
            Tuple of (est_mask, est_pha, *next_states)
        """
        if len(prev_states) != self.num_states:
            raise ValueError(
                f"Expected {self.num_states} states, got {len(prev_states)}"
            )

        # Create state iterator
        state_iter = StateIterator(list(prev_states))

        # Convert to model input format [B, 2, T, F]
        B, F_orig, T = mag.shape
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)

        # Process through encoder with state routing
        x = self._forward_dense_encoder(x, state_iter)

        # After encoder, frequency dimension changes due to stride=(1,2) in dense_conv_2
        # x shape: [B, C, T, F'] where F' = (F_orig + 2) // 2  (approximately)
        _, C, T_enc, F_enc = x.shape

        # Process through TS_BLOCK with state routing
        # Note: We use F_enc (encoded freq dimension) for reshape
        x = self._forward_sequence_block(x, state_iter, B, F_enc)

        # Process through decoders with state routing
        mask = self._forward_mask_decoder(x, state_iter)
        phase_out = self._forward_phase_decoder(x, state_iter)

        # Convert output format
        mask = mask.squeeze(1).transpose(1, 2)  # [B, F, T]

        # Apply mask if masking mode
        if self.infer_type == "masking":
            est_mask = mask
        else:
            est_mask = mask

        if self.phase_output_mode == "complex":
            # phase_out is (x_r, x_i) tuple
            x_r, x_i = phase_out
            x_r = x_r.squeeze(1).transpose(1, 2)  # [B, F, T]
            x_i = x_i.squeeze(1).transpose(1, 2)  # [B, F, T]
            return (est_mask, x_r, x_i) + tuple(state_iter.next_states)
        else:
            # phase_out is atan2 result
            est_pha = phase_out.squeeze(1).transpose(1, 2)  # [B, F, T]
            return (est_mask, est_pha) + tuple(state_iter.next_states)

    def _forward_dense_encoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through DenseEncoder with state routing."""
        # DenseEncoder structure:
        # - dense_conv_1: Conv2d + BatchNorm + PReLU (no state)
        # - dense_block: DS_DDB (multiple AsymmetricConv2d with state)
        # - dense_conv_2: Conv2d + BatchNorm + PReLU (no state)

        encoder = self.dense_encoder

        # dense_conv_1 (stateless)
        x = encoder.dense_conv_1(x)

        # dense_block (stateful DS_DDB)
        x = self._forward_ds_ddb(x, encoder.dense_block, state_iter)

        # dense_conv_2 (stateless)
        x = encoder.dense_conv_2(x)

        return x

    def _forward_ds_ddb(
        self,
        x: Tensor,
        ds_ddb: nn.Module,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through DS_DDB (Dense Dilated Depthwise Block) with state.

        DS_DDB structure:
            skip = x
            for i in depth:
                x = dense_block[i](skip)  # Input is skip, not x!
                skip = cat([x, skip])
            return x

        Note: Each dense_block[i] expects input channels = dense_channel * (i+1)
        because skip grows via concatenation.
        """
        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv2d,
        )

        skip = x
        for i, dense_conv in enumerate(ds_ddb.dense_block):
            # dense_conv is nn.Sequential: [FunctionalConv2d, Conv2d, BatchNorm, PReLU]
            # Input to dense_conv is skip (accumulated features), not x
            layer_input = skip
            for layer in dense_conv:
                if isinstance(layer, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                    # Stateful conv - need state
                    prev_state = state_iter._prev_states[state_iter._idx]
                    layer_input, next_state = layer(
                        layer_input,
                        prev_state,
                        state_frames=self.state_frames_for_update,
                    )
                    state_iter._next_states.append(next_state)
                    state_iter._idx += 1
                else:
                    # Other layers (Conv2d, BatchNorm, PReLU)
                    layer_input = layer(layer_input)
            x = layer_input  # Output of this dense_conv
            skip = torch.cat([x, skip], dim=1)

        return x

    def _forward_sequence_block(
        self,
        x: Tensor,
        state_iter: StateIterator,
        B: int,
        F_dim: int,
    ) -> Tensor:
        """Forward through TS_BLOCK sequence with state routing."""
        for ts_block in self.sequence_block:
            x = self._forward_ts_block(x, ts_block, state_iter, B, F_dim)
        return x

    def _forward_ts_block(
        self,
        x: Tensor,
        ts_block: nn.Module,
        state_iter: StateIterator,
        B: int,
        F_enc: int,
    ) -> Tensor:
        """Forward through single TS_BLOCK with state routing.

        Note: F_enc is the encoded frequency dimension (after encoder stride).
        We use the actual tensor shape for T since it may vary.
        """
        C = ts_block.dense_channel
        _, _, T, F_actual = x.shape
        # Note: Avoid Python asserts here because ONNX tracing may treat them as constants.
        # Shape mismatches should be caught by upstream preprocessing/export configuration.

        # Reshape for time processing: [B, C, T, F] -> [B*F, C, T]
        x = x.permute(0, 3, 1, 2).reshape(B * F_enc, C, T)

        # Time stage
        x = self._forward_stage(x, ts_block.time_stage, state_iter) + x * ts_block.beta_t

        # Reshape for freq processing: [B*F, C, T] -> [B*T, C, F]
        x = x.view(B, F_enc, C, T).permute(0, 3, 2, 1).reshape(B * T, C, F_enc)

        # Freq stage
        x = self._forward_stage(x, ts_block.freq_stage, state_iter) + x * ts_block.beta_f

        # Reshape back: [B*T, C, F] -> [B, C, T, F]
        x = x.view(B, T, C, F_enc).permute(0, 2, 1, 3)

        return x

    def _forward_stage(
        self,
        x: Tensor,
        stage: nn.Module,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through time_stage or freq_stage."""
        for block in stage:
            # Each block is nn.Sequential(CAB, GPKFFN)
            for sub_block in block:
                if hasattr(sub_block, 'sca'):
                    # Channel_Attention_Block
                    x = self._forward_cab(x, sub_block, state_iter)
                elif hasattr(sub_block, 'proj_first'):
                    # Group_Prime_Kernel_FFN
                    x = self._forward_gpkffn(x, sub_block, state_iter)
                else:
                    x = sub_block(x)
        return x

    def _forward_cab(
        self,
        x: Tensor,
        cab: nn.Module,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through Channel_Attention_Block with state."""
        from src.models.onnx_export.layers.functional_stateful import FunctionalStatefulConv1d

        skip = x
        x = cab.norm(x)
        x = cab.pwconv1(x)

        # dwconv - may be FunctionalStatefulConv1d
        if isinstance(cab.dwconv, FunctionalStatefulConv1d):
            prev_state = state_iter._prev_states[state_iter._idx]
            x, next_state = cab.dwconv(
                x,
                prev_state,
                state_frames=self.state_frames_for_update,
            )
            state_iter._next_states.append(next_state)
            state_iter._idx += 1
        else:
            x = cab.dwconv(x)

        x = cab.sg(x)

        # SCA - Sequential with possible FunctionalStatefulConv1d (causal depthwise)
        if isinstance(cab.sca, nn.Sequential):
            sca_out = x
            for layer in cab.sca:
                if isinstance(layer, FunctionalStatefulConv1d):
                    prev_state = state_iter._prev_states[state_iter._idx]
                    sca_out, next_state = layer(
                        sca_out,
                        prev_state,
                        state_frames=self.state_frames_for_update,
                    )
                    state_iter._next_states.append(next_state)
                    state_iter._idx += 1
                else:
                    sca_out = layer(sca_out)
            x = x * sca_out
        else:
            x = x * cab.sca(x)

        x = cab.pwconv2(x)
        x = skip + x * cab.beta

        return x

    def _forward_gpkffn(
        self,
        x: Tensor,
        gpkffn: nn.Module,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through Group_Prime_Kernel_FFN with state."""
        from src.models.onnx_export.layers.functional_stateful import FunctionalStatefulConv1d

        shortcut = x
        x = gpkffn.norm(x)
        x = gpkffn.proj_first(x)

        expand_ratio = gpkffn.expand_ratio
        kernel_list = gpkffn.kernel_list

        x_chunks = list(torch.chunk(x, expand_ratio, dim=1))
        for i in range(expand_ratio):
            kernel_size = kernel_list[i]
            attn_module = getattr(gpkffn, f"attn_{kernel_size}")
            conv_module = getattr(gpkffn, f"conv_{kernel_size}")

            # Process attn path
            attn_out = x_chunks[i]
            for layer in attn_module:
                if isinstance(layer, FunctionalStatefulConv1d):
                    prev_state = state_iter._prev_states[state_iter._idx]
                    attn_out, next_state = layer(
                        attn_out,
                        prev_state,
                        state_frames=self.state_frames_for_update,
                    )
                    state_iter._next_states.append(next_state)
                    state_iter._idx += 1
                else:
                    attn_out = layer(attn_out)

            # Process conv path
            conv_out = x_chunks[i]
            if isinstance(conv_module, FunctionalStatefulConv1d):
                prev_state = state_iter._prev_states[state_iter._idx]
                conv_out, next_state = conv_module(
                    conv_out,
                    prev_state,
                    state_frames=self.state_frames_for_update,
                )
                state_iter._next_states.append(next_state)
                state_iter._idx += 1
            else:
                conv_out = conv_module(conv_out)

            x_chunks[i] = attn_out * conv_out

        x = torch.cat(x_chunks, dim=1)
        x = gpkffn.proj_last(x) * gpkffn.scale + shortcut

        return x

    def _forward_mask_decoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through MaskDecoder with state."""
        decoder = self.mask_decoder

        # dense_block
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)

        # mask_conv (no state)
        x = decoder.mask_conv(x)

        # lsigmoid
        x = x.squeeze(1).transpose(1, 2)
        x = decoder.lsigmoid(x).transpose(1, 2).unsqueeze(1)

        return x

    def _forward_phase_decoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward through PhaseDecoder with state.

        Returns:
            If phase_output_mode == "atan2": phase tensor [B, 1, T, F]
            If phase_output_mode == "complex": tuple (real, imag) each [B, 1, T, F]
        """
        decoder = self.phase_decoder

        # dense_block
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)

        # phase_conv (no state)
        x = decoder.phase_conv(x)
        x_r = decoder.phase_conv_r(x)
        x_i = decoder.phase_conv_i(x)

        if self.phase_output_mode == "complex":
            # Return (real, imag) for host-side atan2 computation
            # This avoids INT8 quantization affecting atan2 precision
            return x_r, x_i
        else:
            # Default: compute atan2 in graph (backwards compatible)
            x = torch.atan2(x_i + 1e-8, x_r + 1e-8)
            return x

    @classmethod
    def from_backbone(
        cls,
        model: nn.Module,
        convert_to_functional: bool = True,
        phase_output_mode: str = "atan2",
    ) -> "StatefulExportableNNCore":
        """
        Create StatefulExportableNNCore from a Backbone model.

        This method:
        1. Converts ConvTranspose2d to ConvTranspose2dWrapper
        2. Converts StatefulConv to FunctionalStateful

        Args:
            model: Backbone model (preferably already converted to stateful)
            convert_to_functional: If True, convert all layers to functional versions
            phase_output_mode: Phase output mode ("atan2" or "complex")
                - "atan2": Compute phase in graph (backwards compatible)
                - "complex": Output (real, imag) for host-side atan2 (recommended for INT8)

        Returns:
            StatefulExportableNNCore instance
        """
        import copy

        model = copy.deepcopy(model)
        model.eval()

        if convert_to_functional:
            # Convert ConvTranspose2d to wrapper
            from src.models.onnx_export.layers.conv_transpose_wrapper import (
                convert_conv_transpose_to_wrapper,
            )
            model, tconv_count = convert_conv_transpose_to_wrapper(model, inplace=True)
            logger.info(f"Converted {tconv_count} ConvTranspose2d to wrapper")

            # Convert StatefulConv to FunctionalStateful
            conv_count = convert_stateful_to_functional(model)
            logger.info(f"Converted {conv_count} StatefulConv to functional")

        # Extract model config
        n_fft = getattr(model, 'fft_len', 400)
        infer_type = getattr(model, 'infer_type', 'masking')

        core = cls(
            dense_encoder=model.dense_encoder,
            sequence_block=model.sequence_block,
            mask_decoder=model.mask_decoder,
            phase_decoder=model.phase_decoder,
            infer_type=infer_type,
            n_fft=n_fft,
            phase_output_mode=phase_output_mode,
        )

        return core


def convert_stateful_to_functional(model: nn.Module) -> int:
    """
    Convert all StatefulConv layers to FunctionalStateful in-place.

    Args:
        model: Model to convert

    Returns:
        Number of converted layers
    """
    from src.models.onnx_export.layers.functional_stateful import convert_to_functional
    from src.models.streaming.layers.stateful_conv import (
        StatefulAsymmetricConv2d,
        StatefulCausalConv1d,
        StatefulCausalConv2d,
    )

    stateful_types = (
        StatefulCausalConv1d,
        StatefulAsymmetricConv2d,
        StatefulCausalConv2d,
    )

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, stateful_types):
            # Find parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            # Convert and replace
            functional = convert_to_functional(module)
            attr_name = parts[-1]
            if attr_name.isdigit():
                parent[int(attr_name)] = functional
            else:
                setattr(parent, attr_name, functional)
            count += 1

    return count


def export_stateful_nncore_to_onnx(
    core: StatefulExportableNNCore,
    output_path: str,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    opset_version: int = 17,
    verbose: bool = True,
    use_dynamic_axes: bool = True,
) -> str:
    """
    Export StatefulExportableNNCore to ONNX format.

    This exports the full stateful model with all state tensors as
    explicit graph inputs and outputs.

    Args:
        core: StatefulExportableNNCore instance
        output_path: Path for output ONNX file
        batch_size: Batch size for export
        time_frames: Number of time frames
        freq_size: Frequency dimension size
        opset_version: ONNX opset version
        verbose: Print export info
        use_dynamic_axes: If True, use dynamic axes for batch/time.
            Set to False for INT8 quantization (static shapes required).

    Returns:
        Path to exported ONNX file
    """
    import os

    core.eval()
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Create dummy inputs
    mag = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)
    pha = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)
    states = core.init_states(batch_size, freq_size, time_frames, device, dtype)

    # Build input/output names based on phase_output_mode
    input_names = ["mag", "pha"] + core.get_state_names()

    if core.phase_output_mode == "complex":
        # Output (est_mask, phase_real, phase_imag) + states
        output_names = ["est_mask", "phase_real", "phase_imag"] + [f"next_{name}" for name in core.get_state_names()]
    else:
        # Output (est_mask, est_pha) + states
        output_names = ["est_mask", "est_pha"] + [f"next_{name}" for name in core.get_state_names()]

    # Dynamic axes for streaming flexibility
    if use_dynamic_axes:
        dynamic_axes = {
            "mag": {0: "batch", 2: "time"},
            "pha": {0: "batch", 2: "time"},
            "est_mask": {0: "batch", 2: "time"},
        }
        if core.phase_output_mode == "complex":
            dynamic_axes["phase_real"] = {0: "batch", 2: "time"}
            dynamic_axes["phase_imag"] = {0: "batch", 2: "time"}
        else:
            dynamic_axes["est_pha"] = {0: "batch", 2: "time"}

        # Add dynamic axes for states.
        # IMPORTANT:
        # In TS_BLOCK, effective batch differs by stage due to reshape:
        # - time_stage: B*F_enc
        # - freq_stage: B*T
        # Forcing all states to share the same symbolic "batch" axis can create
        # invalid equality constraints in ONNX Runtime (shape mismatch warnings).
        # We therefore use distinct symbols per stage family.
        for state_name in core.get_state_names():
            if "time_stage" in state_name:
                bsym = "batch_time_stage"
            elif "freq_stage" in state_name:
                bsym = "batch_freq_stage"
            else:
                bsym = "batch"
            dynamic_axes[state_name] = {0: bsym}
            dynamic_axes[f"next_{state_name}"] = {0: bsym}
    else:
        # Static shapes for INT8 quantization
        dynamic_axes = None

    if verbose:
        print(f"Exporting stateful core to: {output_path}")
        print(f"  Input shape: mag/pha [{batch_size}, {freq_size}, {time_frames}]")
        print(f"  Number of states: {core.num_states}")
        print(f"  Phase output mode: {core.phase_output_mode}")
        print(f"  Shape mode: {'dynamic' if use_dynamic_axes else 'static'}")
        print(f"  Opset version: {opset_version}")

    # Create parent directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Prepare inputs tuple
    inputs = (mag, pha) + tuple(states)

    torch.onnx.export(
        core,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        # Torch 2.9+ defaults to the Dynamo exporter (dynamo=True) which depends on onnxscript.
        # Use the legacy exporter for maximal environment compatibility.
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Exported: {file_size:.2f} MB")

    return output_path


def verify_stateful_onnx_export(
    onnx_path: str,
    core: StatefulExportableNNCore,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    num_steps: int = 5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Verify stateful ONNX export against PyTorch model over multiple steps.

    This verifies that the ONNX model produces the same outputs as PyTorch
    when run for multiple steps with state propagation.

    Args:
        onnx_path: Path to ONNX file
        core: Original StatefulExportableNNCore
        batch_size: Batch size for test
        time_frames: Number of time frames per step
        freq_size: Frequency dimension
        num_steps: Number of steps to run
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with verification results
    """
    from .verify_utils import verify_stateful_onnx_multistep

    return verify_stateful_onnx_multistep(
        onnx_path=onnx_path,
        core=core,
        init_states_fn=core.init_states,
        get_state_names_fn=core.get_state_names,
        num_non_state_outputs=2,
        batch_size=batch_size,
        time_frames=time_frames,
        freq_size=freq_size,
        num_steps=num_steps,
        atol=atol,
        rtol=rtol,
    )


__all__ = [
    "StatefulExportableNNCore",
    "StateIterator",
    "StateSpec",
    "convert_stateful_to_functional",
    "export_stateful_nncore_to_onnx",
    "verify_stateful_onnx_export",
]
