"""
Stateful Reshape-Free Exportable Neural Network Core for ONNX.

This module provides a monolithic ONNX-exportable wrapper that uses reshape-free
TS_BLOCK layers, eliminating the B*F/B*T reshape overhead present in the original
StatefulExportableNNCore.

Key differences from StatefulExportableNNCore:
1. TS_BLOCK uses StatefulReshapeFreeTSBlock (4D tensors, no reshape)
2. RF states are managed with flat tensor lists (not Dict/List nesting)
3. Encoder/Decoder still use the original functional stateful layers
4. All state dynamic axes use a single "batch" symbol (no B*F/B*T distinction)

Graph boundary (same as stateful_core.py):
    Host FP32:  audio -> STFT -> mag/pha
    ONNX:       mag/pha + prev_states -> NNCore -> est_mask/est_pha + next_states
    Host FP32:  est_mask/est_pha -> complex -> iSTFT -> audio
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.onnx_export.stateful_core import (
    StateIterator,
    convert_stateful_to_functional,
)

logger = logging.getLogger(__name__)


class StatefulReshapeFreeExportableNNCore(nn.Module):
    """
    Fully stateful ONNX-exportable neural network core with reshape-free TS_BLOCK.

    Architecture:
        mag/pha -> DenseEncoder (functional stateful)
                -> TS_BLOCK x N (reshape-free stateful, 4D)
                -> MaskDecoder/PhaseDecoder (functional stateful)

    State partitioning:
        [encoder_states...] + [rf_states_flat...] + [decoder_states...]

    Encoder/Decoder states use the existing functional stateful layer mechanism
    (StateIterator). RF states are flattened from the nested Dict/List structure
    of StatefulReshapeFreeTSBlock into a flat tensor list for ONNX I/O.

    Input/Output:
        Inputs:
            - mag: Magnitude spectrogram [B, F, T]
            - pha: Phase spectrogram [B, F, T]
            - *prev_states: All previous state tensors

        Outputs:
            - est_mask: Estimated mask [B, F, T]
            - est_pha or (phase_real, phase_imag): Phase output [B, F, T]
            - *next_states: All updated state tensors
    """

    def __init__(
        self,
        dense_encoder: nn.Module,
        rf_sequence_block: nn.ModuleList,
        mask_decoder: nn.Module,
        phase_decoder: nn.Module,
        infer_type: str = "masking",
        n_fft: int = 400,
        phase_output_mode: str = "atan2",
    ):
        """
        Initialize StatefulReshapeFreeExportableNNCore.

        Note: Use from_backbone() for easier creation.

        Args:
            dense_encoder: DenseEncoder (converted to functional layers)
            rf_sequence_block: ModuleList of StatefulReshapeFreeTSBlock
            mask_decoder: MaskDecoder (converted to functional layers)
            phase_decoder: PhaseDecoder (converted to functional layers)
            infer_type: "masking" or "mapping"
            n_fft: FFT size for frequency dimension calculation
            phase_output_mode: "atan2" or "complex"
        """
        super().__init__()
        self.dense_encoder = dense_encoder
        self.rf_sequence_block = rf_sequence_block
        self.mask_decoder = mask_decoder
        self.phase_decoder = phase_decoder
        self.infer_type = infer_type
        self.n_fft = n_fft
        self.phase_output_mode = phase_output_mode

        # State frames gating for extended (lookahead) inputs
        self.state_frames_for_update: Optional[int] = None

        # Collect functional modules for encoder/decoder only
        # (RF sequence block manages its own states)
        self._enc_functional_modules: List[Tuple[str, nn.Module]] = []
        self._dec_functional_modules: List[Tuple[str, nn.Module]] = []
        self._collect_functional_modules()

        # Compute RF state layout
        self._rf_state_layout: List[List[str]] = []  # Per-block list of state key names
        self._rf_states_per_block: List[int] = []
        self._total_rf_states: int = 0
        self._compute_rf_state_layout()

    def _collect_functional_modules(self) -> None:
        """Collect functional stateful modules for encoder and decoder only.

        Unlike StatefulExportableNNCore which traverses self.named_modules(),
        we only traverse encoder and decoder submodules since the RF sequence
        block manages its own state through the reshape-free state interface.
        """
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

        # Encoder modules
        self._enc_functional_modules = []
        enc_collected = set()
        for name, module in self.dense_encoder.named_modules():
            full_name = f"dense_encoder.{name}" if name else "dense_encoder"
            if isinstance(module, functional_types):
                is_nested = any(full_name.startswith(p + '.') for p in enc_collected)
                if not is_nested:
                    self._enc_functional_modules.append((full_name, module))
                    enc_collected.add(full_name)

        # Decoder modules (mask + phase)
        self._dec_functional_modules = []
        dec_collected = set()
        for decoder_attr in ("mask_decoder", "phase_decoder"):
            decoder = getattr(self, decoder_attr)
            for name, module in decoder.named_modules():
                full_name = f"{decoder_attr}.{name}" if name else decoder_attr
                if isinstance(module, functional_types):
                    is_nested = any(full_name.startswith(p + '.') for p in dec_collected)
                    if not is_nested:
                        self._dec_functional_modules.append((full_name, module))
                        dec_collected.add(full_name)

        logger.info(
            f"Collected {len(self._enc_functional_modules)} encoder + "
            f"{len(self._dec_functional_modules)} decoder functional modules"
        )

    def _compute_rf_state_layout(self) -> None:
        """Compute the flat state layout for reshape-free TS_BLOCKs.

        Each StatefulReshapeFreeTSBlock has states organized as:
            List[Dict[str, Dict[str, Tensor]]]
            = [{"cab": {"dwconv": ..., "sca_dwconv": ...}, "gpkffn": {"attn_dw_3": ..., ...}}, ...]

        We flatten this to a deterministic ordered list of tensor keys per block:
            [cab_0_dwconv, cab_0_sca_dwconv, gpkffn_0_attn_dw_3, gpkffn_0_conv_3, ...,
             cab_1_dwconv, cab_1_sca_dwconv, gpkffn_1_attn_dw_3, ...]
        """
        self._rf_state_layout = []
        self._rf_states_per_block = []
        self._total_rf_states = 0

        for block_idx, rf_block in enumerate(self.rf_sequence_block):
            block_keys = []
            for tb_idx in range(rf_block.time_block_num):
                # CAB states (deterministic order)
                block_keys.append(f"tb{tb_idx}_cab_dwconv")
                block_keys.append(f"tb{tb_idx}_cab_sca_dwconv")

                # GPKFFN states (sorted by kernel for determinism)
                gpkffn = rf_block.time_gpkffns[tb_idx]
                for k in sorted(gpkffn.kernel_list):
                    block_keys.append(f"tb{tb_idx}_gpkffn_attn_dw_{k}")
                    block_keys.append(f"tb{tb_idx}_gpkffn_conv_{k}")

            self._rf_state_layout.append(block_keys)
            self._rf_states_per_block.append(len(block_keys))
            self._total_rf_states += len(block_keys)

        logger.info(
            f"RF state layout: {len(self.rf_sequence_block)} blocks, "
            f"{self._total_rf_states} total states"
        )

    def _flatten_rf_states(self, nested: List[Dict[str, Any]], block_idx: int) -> List[Tensor]:
        """Flatten nested state dicts from a single RF block to a flat tensor list.

        Args:
            nested: List of state dicts from StatefulReshapeFreeTSBlock.forward()
                    [{"cab": {"dwconv": T, "ema": T}, "gpkffn": {"attn_dw_3": T, ...}}, ...]
            block_idx: Index of the RF block (for layout lookup)

        Returns:
            Flat list of state tensors in deterministic order
        """
        flat = []
        keys = self._rf_state_layout[block_idx]
        rf_block = self.rf_sequence_block[block_idx]

        key_idx = 0
        for tb_idx in range(rf_block.time_block_num):
            cab_state = nested[tb_idx]["cab"]
            gpkffn_state = nested[tb_idx]["gpkffn"]

            # CAB: dwconv, sca_dwconv
            flat.append(cab_state["dwconv"])
            key_idx += 1
            flat.append(cab_state["sca_dwconv"])
            key_idx += 1

            # GPKFFN: sorted kernels
            gpkffn = rf_block.time_gpkffns[tb_idx]
            for k in sorted(gpkffn.kernel_list):
                flat.append(gpkffn_state[f"attn_dw_{k}"])
                key_idx += 1
                flat.append(gpkffn_state[f"conv_{k}"])
                key_idx += 1

        return flat

    def _unflatten_rf_states(self, flat: List[Tensor], block_idx: int) -> List[Dict[str, Any]]:
        """Unflatten a flat tensor list back to nested state dicts.

        Args:
            flat: Flat list of state tensors
            block_idx: Index of the RF block (for layout lookup)

        Returns:
            Nested state dicts matching StatefulReshapeFreeTSBlock.forward() input format
        """
        rf_block = self.rf_sequence_block[block_idx]
        nested = []
        idx = 0

        for tb_idx in range(rf_block.time_block_num):
            cab_state = {}
            gpkffn_state = {}

            # CAB: dwconv, sca_dwconv
            cab_state["dwconv"] = flat[idx]
            idx += 1
            cab_state["sca_dwconv"] = flat[idx]
            idx += 1

            # GPKFFN: sorted kernels
            gpkffn = rf_block.time_gpkffns[tb_idx]
            for k in sorted(gpkffn.kernel_list):
                gpkffn_state[f"attn_dw_{k}"] = flat[idx]
                idx += 1
                gpkffn_state[f"conv_{k}"] = flat[idx]
                idx += 1

            nested.append({"cab": cab_state, "gpkffn": gpkffn_state})

        return nested

    @property
    def num_enc_states(self) -> int:
        return len(self._enc_functional_modules)

    @property
    def num_dec_states(self) -> int:
        return len(self._dec_functional_modules)

    @property
    def num_rf_states(self) -> int:
        return self._total_rf_states

    @property
    def num_states(self) -> int:
        return self.num_enc_states + self.num_rf_states + self.num_dec_states

    def get_state_names(self) -> List[str]:
        """Get names of all state tensors in partition order."""
        names = []

        # Encoder states
        for i, (mod_name, _) in enumerate(self._enc_functional_modules):
            names.append(f"state_enc_{i}_{mod_name.replace('.', '_')}")

        # RF states
        for block_idx, block_keys in enumerate(self._rf_state_layout):
            for key in block_keys:
                names.append(f"state_rf_{block_idx}_{key}")

        # Decoder states
        for i, (mod_name, _) in enumerate(self._dec_functional_modules):
            names.append(f"state_dec_{i}_{mod_name.replace('.', '_')}")

        return names

    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Set a fixed number of frames for state updates.

        Args:
            state_frames: Number of frames allowed to update state.
                If None, all frames update state (default).
        """
        self.state_frames_for_update = int(state_frames) if state_frames is not None else None

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
            time_frames: Number of time frames (for freq_stage batch dim)
            device: Device for states
            dtype: Data type for states

        Returns:
            List of initialized state tensors in partition order
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

        freq_enc = (freq_size - 1) // 2

        states = []

        # --- Encoder states ---
        for name, module in self._enc_functional_modules:
            if "dense_encoder" in name:
                conv2d_freq = freq_size
            else:
                conv2d_freq = freq_enc

            if isinstance(module, FunctionalStatefulConv1d):
                if "time_stage" in name:
                    effective_batch = batch_size * freq_enc
                elif "freq_stage" in name:
                    effective_batch = batch_size * time_frames
                else:
                    effective_batch = batch_size
                state = module.init_state(effective_batch, device, dtype)
            elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                state = module.init_state(batch_size, conv2d_freq, device, dtype)
            else:
                raise TypeError(f"Unknown encoder module type: {type(module)}")
            states.append(state)

        # --- RF states (reshape-free TS_BLOCK) ---
        for block_idx, rf_block in enumerate(self.rf_sequence_block):
            # RF states use B (not B*F or B*T) since no reshape is needed
            nested_states = rf_block.init_state(batch_size, freq_enc, device, dtype)
            flat_states = self._flatten_rf_states(nested_states, block_idx)
            states.extend(flat_states)

        # --- Decoder states ---
        for name, module in self._dec_functional_modules:
            if "dense_encoder" in name:
                conv2d_freq = freq_size
            else:
                conv2d_freq = freq_enc

            if isinstance(module, FunctionalStatefulConv1d):
                if "time_stage" in name:
                    effective_batch = batch_size * freq_enc
                elif "freq_stage" in name:
                    effective_batch = batch_size * time_frames
                else:
                    effective_batch = batch_size
                state = module.init_state(effective_batch, device, dtype)
            elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                state = module.init_state(batch_size, conv2d_freq, device, dtype)
            else:
                raise TypeError(f"Unknown decoder module type: {type(module)}")
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

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]
            *prev_states: Previous state tensors (must match num_states)

        Returns:
            Tuple of (est_mask, est_pha_or_complex, *next_states)
        """
        if len(prev_states) != self.num_states:
            raise ValueError(
                f"Expected {self.num_states} states, got {len(prev_states)}"
            )

        # Partition states
        n_enc = self.num_enc_states
        n_rf = self.num_rf_states
        enc_states = list(prev_states[:n_enc])
        rf_states_flat = list(prev_states[n_enc:n_enc + n_rf])
        dec_states = list(prev_states[n_enc + n_rf:])

        # Convert to model input format [B, 2, T, F]
        B, F_orig, T = mag.shape
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)

        # --- Encoder ---
        enc_iter = StateIterator(enc_states)
        x = self._forward_dense_encoder(x, enc_iter)
        enc_next = enc_iter.next_states

        # x shape: [B, C, T, F_enc]

        # --- RF Sequence Block ---
        x, rf_next_flat = self._forward_rf_sequence_block(x, rf_states_flat)

        # --- Decoders ---
        dec_iter = StateIterator(dec_states)
        mask = self._forward_mask_decoder(x, dec_iter)
        phase_out = self._forward_phase_decoder(x, dec_iter)
        dec_next = dec_iter.next_states

        # Convert output format
        mask = mask.squeeze(1).transpose(1, 2)  # [B, F, T]

        if self.infer_type == "masking":
            est_mask = mask
        else:
            est_mask = mask

        if self.phase_output_mode == "complex":
            x_r, x_i = phase_out
            x_r = x_r.squeeze(1).transpose(1, 2)  # [B, F, T]
            x_i = x_i.squeeze(1).transpose(1, 2)  # [B, F, T]
            return (est_mask, x_r, x_i) + tuple(enc_next) + tuple(rf_next_flat) + tuple(dec_next)
        else:
            est_pha = phase_out.squeeze(1).transpose(1, 2)  # [B, F, T]
            return (est_mask, est_pha) + tuple(enc_next) + tuple(rf_next_flat) + tuple(dec_next)

    # =========================================================================
    # Encoder/Decoder forward methods (reused from StatefulExportableNNCore)
    # =========================================================================

    def _forward_dense_encoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through DenseEncoder with state routing."""
        encoder = self.dense_encoder
        x = encoder.dense_conv_1(x)
        x = self._forward_ds_ddb(x, encoder.dense_block, state_iter)
        x = encoder.dense_conv_2(x)
        return x

    def _forward_ds_ddb(
        self,
        x: Tensor,
        ds_ddb: nn.Module,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through DS_DDB with state routing."""
        from src.models.onnx_export.layers.functional_stateful import (
            FunctionalStatefulCausalConv2d,
            FunctionalStatefulConv2d,
        )

        skip = x
        for i, dense_conv in enumerate(ds_ddb.dense_block):
            layer_input = skip
            for layer in dense_conv:
                if isinstance(layer, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                    prev_state = state_iter._prev_states[state_iter._idx]
                    layer_input, next_state = layer(
                        layer_input,
                        prev_state,
                        state_frames=self.state_frames_for_update,
                    )
                    state_iter._next_states.append(next_state)
                    state_iter._idx += 1
                else:
                    layer_input = layer(layer_input)
            x = layer_input
            skip = torch.cat([x, skip], dim=1)

        return x

    def _forward_mask_decoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Tensor:
        """Forward through MaskDecoder with state."""
        decoder = self.mask_decoder
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)
        x = decoder.mask_conv(x)
        x = x.squeeze(1).transpose(1, 2)
        x = decoder.lsigmoid(x).transpose(1, 2).unsqueeze(1)
        return x

    def _forward_phase_decoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward through PhaseDecoder with state."""
        decoder = self.phase_decoder
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)
        x = decoder.phase_conv(x)
        x_r = decoder.phase_conv_r(x)
        x_i = decoder.phase_conv_i(x)

        if self.phase_output_mode == "complex":
            return x_r, x_i
        else:
            x = torch.atan2(x_i + 1e-8, x_r + 1e-8)
            return x

    # =========================================================================
    # RF Sequence Block forward
    # =========================================================================

    def _forward_rf_sequence_block(
        self,
        x: Tensor,
        rf_states_flat: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        """Forward through reshape-free TS_BLOCKs with flat state I/O.

        Args:
            x: Input tensor [B, C, T, F_enc]
            rf_states_flat: Flat list of all RF state tensors

        Returns:
            Tuple of (output, next_rf_states_flat)
        """
        next_flat_all = []
        offset = 0

        for block_idx, rf_block in enumerate(self.rf_sequence_block):
            n_states = self._rf_states_per_block[block_idx]
            block_flat = rf_states_flat[offset:offset + n_states]
            offset += n_states

            # Unflatten to nested dict structure
            block_nested = self._unflatten_rf_states(block_flat, block_idx)

            # Forward through reshape-free TS_BLOCK
            x, new_nested = rf_block(x, block_nested, state_frames=self.state_frames_for_update)

            # Flatten back
            new_flat = self._flatten_rf_states(new_nested, block_idx)
            next_flat_all.extend(new_flat)

        return x, next_flat_all

    # =========================================================================
    # Factory method
    # =========================================================================

    @classmethod
    def from_backbone(
        cls,
        model: nn.Module,
        convert_to_functional: bool = True,
        phase_output_mode: str = "atan2",
    ) -> "StatefulReshapeFreeExportableNNCore":
        """
        Create StatefulReshapeFreeExportableNNCore from a Backbone model.

        Conversion pipeline:
        1. ConvTranspose2d -> ConvTranspose2dWrapper (full model)
        2. StatefulConv -> FunctionalStateful (full model)
        3. TS_BLOCK sequence -> StatefulReshapeFreeTSBlock (replaces sequence_block)

        Step 3 reads weights from FunctionalStatefulConv1d.conv (created in step 2),
        so it must come after step 2.

        Args:
            model: Backbone model (preferably already converted to stateful)
            convert_to_functional: If True, convert all layers to functional versions
            phase_output_mode: "atan2" or "complex"

        Returns:
            StatefulReshapeFreeExportableNNCore instance
        """
        from src.models.streaming.converters.reshape_free_converter import (
            convert_sequence_block_to_stateful_reshape_free,
        )

        model = copy.deepcopy(model)
        model.eval()

        if convert_to_functional:
            # Step 1: ConvTranspose2d -> wrapper
            from src.models.onnx_export.layers.conv_transpose_wrapper import (
                convert_conv_transpose_to_wrapper,
            )
            model, tconv_count = convert_conv_transpose_to_wrapper(model, inplace=True)
            logger.info(f"Converted {tconv_count} ConvTranspose2d to wrapper")

            # Step 2: StatefulConv -> FunctionalStateful (full model including TS_BLOCK)
            conv_count = convert_stateful_to_functional(model)
            logger.info(f"Converted {conv_count} StatefulConv to functional")

        # Step 3: Convert sequence_block to reshape-free stateful.
        # The converter reads FunctionalStatefulConv1d.conv weights (from step 2).
        rf_sequence_block = convert_sequence_block_to_stateful_reshape_free(
            model.sequence_block,
        )
        logger.info(f"Converted {len(rf_sequence_block)} TS_BLOCKs to reshape-free stateful")

        # Extract model config
        n_fft = getattr(model, 'fft_len', 400)
        infer_type = getattr(model, 'infer_type', 'masking')

        core = cls(
            dense_encoder=model.dense_encoder,
            rf_sequence_block=rf_sequence_block,
            mask_decoder=model.mask_decoder,
            phase_decoder=model.phase_decoder,
            infer_type=infer_type,
            n_fft=n_fft,
            phase_output_mode=phase_output_mode,
        )

        return core


# =============================================================================
# Export function
# =============================================================================


def export_stateful_rf_nncore_to_onnx(
    core: StatefulReshapeFreeExportableNNCore,
    output_path: str,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    opset_version: int = 17,
    verbose: bool = True,
    use_dynamic_axes: bool = True,
) -> str:
    """
    Export StatefulReshapeFreeExportableNNCore to ONNX format.

    Args:
        core: StatefulReshapeFreeExportableNNCore instance
        output_path: Path for output ONNX file
        batch_size: Batch size for export
        time_frames: Number of time frames
        freq_size: Frequency dimension size
        opset_version: ONNX opset version (default: 17)
        verbose: Print export info
        use_dynamic_axes: If True, use dynamic axes for batch/time

    Returns:
        Path to exported ONNX file
    """
    core.eval()
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Create dummy inputs
    mag = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)
    pha = torch.randn(batch_size, freq_size, time_frames, device=device, dtype=dtype)
    states = core.init_states(batch_size, freq_size, time_frames, device, dtype)

    # Build input/output names
    input_names = ["mag", "pha"] + core.get_state_names()

    if core.phase_output_mode == "complex":
        output_names = ["est_mask", "phase_real", "phase_imag"] + [
            f"next_{name}" for name in core.get_state_names()
        ]
    else:
        output_names = ["est_mask", "est_pha"] + [
            f"next_{name}" for name in core.get_state_names()
        ]

    # Dynamic axes
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

        # RF states all use "batch" symbol (no B*F/B*T distinction needed)
        for state_name in core.get_state_names():
            dynamic_axes[state_name] = {0: "batch"}
            dynamic_axes[f"next_{state_name}"] = {0: "batch"}
    else:
        dynamic_axes = None

    if verbose:
        print(f"Exporting reshape-free stateful core to: {output_path}")
        print(f"  Input shape: mag/pha [{batch_size}, {freq_size}, {time_frames}]")
        print(f"  Number of states: {core.num_states}")
        print(f"    Encoder: {core.num_enc_states}")
        print(f"    RF TS_BLOCK: {core.num_rf_states}")
        print(f"    Decoder: {core.num_dec_states}")
        print(f"  Phase output mode: {core.phase_output_mode}")
        print(f"  Shape mode: {'dynamic' if use_dynamic_axes else 'static'}")
        print(f"  Opset version: {opset_version}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    inputs = (mag, pha) + tuple(states)

    torch.onnx.export(
        core,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Exported: {file_size:.2f} MB")

    return output_path


# =============================================================================
# Verification function
# =============================================================================


def verify_stateful_rf_onnx_export(
    core: StatefulReshapeFreeExportableNNCore,
    onnx_path: str,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    num_steps: int = 20,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Verify stateful RF ONNX export against PyTorch model over multiple steps.

    Args:
        core: Original StatefulReshapeFreeExportableNNCore
        onnx_path: Path to ONNX file
        batch_size: Batch size for test
        time_frames: Number of time frames per step
        freq_size: Frequency dimension
        num_steps: Number of steps to run (default: 20)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with verification results
    """
    from .verify_utils import verify_stateful_onnx_multistep

    num_non_state_outputs = 3 if core.phase_output_mode == "complex" else 2

    return verify_stateful_onnx_multistep(
        onnx_path=onnx_path,
        core=core,
        init_states_fn=core.init_states,
        get_state_names_fn=core.get_state_names,
        num_non_state_outputs=num_non_state_outputs,
        batch_size=batch_size,
        time_frames=time_frames,
        freq_size=freq_size,
        num_steps=num_steps,
        atol=atol,
        rtol=rtol,
        verbose=True,
    )


__all__ = [
    "StatefulReshapeFreeExportableNNCore",
    "export_stateful_rf_nncore_to_onnx",
    "verify_stateful_rf_onnx_export",
]
