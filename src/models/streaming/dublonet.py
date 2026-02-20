"""
DuBLoNet with Lookahead Buffering.

This module provides a streaming wrapper that addresses the asymmetric
convolution problem by buffering features and providing real lookahead context.
It supports models where encoder, decoder, or both have asymmetric padding.

Processing pipeline:
1. Input buffering + STFT context:
   - Accumulate samples until a full processing window is available.
   - Use an overlap-add friendly STFT configuration (center=True) by prepending
     win_size/2 samples of context from the previous step and discarding the
     corresponding context frames in the STFT output.
2. Encoder + TS_BLOCK:
   - Run the encoder path immediately once the processing window is ready.
   - StatefulConv provides past context via internal state buffers.
   - "Input lookahead" provides future context for asymmetric encoder padding.
3. Feature buffer (decoder lookahead):
   - Accumulate encoder features until enough frames are available for decoder
     processing with lookahead.
4. Decoder:
   - Process an extended time window (current chunk + lookahead + STFT frames)
     and produce only the current chunk output for iSTFT overlap-add.
5. StateFramesContext:
   - Prevent lookahead and STFT-induced extra frames from corrupting streaming
     states by limiting state updates to the current chunk frames.

This approach provides:
- Real past context from StatefulConv buffering
- Real future context from delayed processing (encoder/decoder lookahead)
- Minimal additional latency beyond the required lookahead frames

Example (decoder-only asymmetric):
    >>> streaming = DuBLoNet.from_checkpoint(
    ...     chkpt_dir="results/experiments/prk_1117_1",
    ...     chunk_size=64,
    ...     encoder_lookahead=0,   # Encoder is causal
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )

Example (both encoder and decoder asymmetric):
    >>> streaming = DuBLoNet.from_checkpoint(
    ...     chkpt_dir="results/experiments/prk_1114_2",
    ...     chunk_size=64,
    ...     encoder_lookahead=7,   # Encoder needs 7 frame lookahead
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )
    >>>
    >>> for chunk in audio_stream:
    ...     output = streaming.process_samples(chunk)
    ...     if output is not None:
    ...         play(output)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.stft import mag_pha_istft, mag_pha_stft

if TYPE_CHECKING:
    from src.models.streaming.layers.reshape_free_stateful import (
        StatefulReshapeFreeTSBlock,
    )

logger = logging.getLogger(__name__)


class DuBLoNet(nn.Module):
    """
    Streaming wrapper for Backbone with lookahead buffering.

    This wrapper enables true streaming inference for models with asymmetric
    convolutions in encoder, decoder, or both. It provides real future context
    through lookahead buffering instead of zero-padding.

    Supported model configurations:
    - **Fully Causal**: encoder_lookahead=0, decoder_lookahead=0
    - **Asymmetric Decoder Only**: encoder_lookahead=0, decoder_lookahead>0
      Example: prk_1117_1 with decoder_padding_ratio=(0.77, 0.23)
    - **Asymmetric Encoder+Decoder**: encoder_lookahead>0, decoder_lookahead>0
      Example: prk_1114_2 with both enc/dec having (0.77, 0.23) ratio

    Processing pipeline:

    1. **Input buffering + STFT context**:
       - Accumulates input samples until a full processing window is available.
       - Uses STFT with center=True and prepends win_size/2 samples of context
         from the previous step to keep frame alignment stable across chunk
         boundaries.
       - Even when encoder_lookahead == 0, at least one STFT lookahead frame is
         used for stable overlap-add in streaming.

    2. **Encoder Path**:
       - DenseEncoder + TS_BLOCK with StatefulConv
       - StatefulConv provides past context (left padding replacement)
       - Input lookahead provides future context (right padding replacement for
         asymmetric encoder padding)

    3. **Feature Buffer** (for decoder lookahead):
       - Stores encoded features until decoder_lookahead frames available
       - Bridges encoder output to decoder input with proper context

    4. **Decoder Path** (delayed processing):
       - Processes extended input: current + lookahead frames
       - Uses StateFramesContext to prevent state corruption from lookahead
       - Outputs only current chunk portion (lookahead frames discarded)

    The total latency is measured in frames and includes STFT lookahead:
        total_lookahead_frames = input_lookahead_frames + decoder_lookahead
    where:
        input_lookahead_frames = max(stft_lookahead_frames, encoder_lookahead)
    and stft_lookahead_frames is at least 1 when center=True.
    The corresponding sample/seconds latency is:
        latency_samples = total_lookahead_frames * hop_size
        latency_seconds = latency_samples / sample_rate

    Attributes:
        encoder_lookahead: Frames needed for encoder (0 if encoder is causal)
        decoder_lookahead: Frames needed for decoder (0 if decoder is causal)
        total_lookahead: input_lookahead_frames + decoder_lookahead
        latency_ms: Total latency in milliseconds
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        stft_lookahead_frames: int = 1,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
        rf_sequence_block: Optional[nn.ModuleList] = None,
        freq_size: int = 100,
    ):
        """
        Initialize DuBLoNet.

        Note: Use `from_checkpoint()` for easier initialization.

        Args:
            model: Backbone model (should already have StatefulConv applied)
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames to delay encoder processing (for asymmetric encoder)
            decoder_lookahead: Frames to delay decoder processing (for asymmetric decoder)
            stft_lookahead_frames: Minimum STFT lookahead frames used for stable streaming
                overlap-add with center=True. Larger values increase latency but can
                reduce boundary sensitivity for very small chunk sizes.
            hop_size: STFT hop size in samples
            n_fft: FFT size
            win_size: Window size
            compress_factor: Magnitude compression factor
            sample_rate: Audio sample rate in Hz
            rf_sequence_block: Reshape-free TS_BLOCK ModuleList (if enabled)
            freq_size: Frequency bins (for state initialization)
        """
        super().__init__()

        # Store model reference
        self.model = model
        self.model.eval()

        # STFT parameters
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate

        # Streaming parameters
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        # Even when encoder_lookahead == 0, we need at least 1 frame of input lookahead
        # for stable STFT/iSTFT streaming with center=True.
        self.stft_lookahead_frames = max(1, int(stft_lookahead_frames))
        self.input_lookahead_frames = max(self.stft_lookahead_frames, int(encoder_lookahead))
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        # Calculate samples per chunk (input)
        # With center=True STFT (n_fft == win_size), signal_length samples
        # produce (signal_length // hop_size + 1) frames.
        # We need total_frames_needed = chunk_size + input_lookahead_frames.
        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size

        # Output step is ALWAYS chunk_size frames (i.e., chunk_size * hop_size samples).
        # This must match how we slide the input buffer; otherwise, time alignment drifts.
        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = self.output_frames_per_chunk * hop_size

        # Latency calculation
        self.latency_samples = self.total_lookahead * hop_size
        self.latency_ms = self.latency_samples / sample_rate * 1000

        # Reshape-free TS_BLOCK support
        self.rf_sequence_block = rf_sequence_block
        self.use_reshape_free = rf_sequence_block is not None
        self.freq_size = freq_size
        self._rf_states: Optional[List[List[Dict[str, Tensor]]]] = None

        # Initialize buffers
        self._reset_buffers()

        # Config storage
        self._streaming_config: Dict[str, Any] = {}

    def _reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self.feature_buffer: List[Dict[str, Any]] = []  # Encoded features for decoder lookahead
        self._buffered_frames = 0
        # STFT context: last win_size/2 samples from previous chunk for proper center=True handling.
        # Initialize with zeros (silence before audio starts) so the first chunk also gets
        # context prepended, avoiding reflect-padding failures when samples_per_chunk <= n_fft//2.
        self._stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)
        self._stft_context_frames = self.win_size // (2 * self.hop_size)  # frames to skip after prepending

        # Initialize reshape-free states if enabled
        if self.use_reshape_free and self.rf_sequence_block is not None:
            self._rf_states = self._init_rf_states()

    def _init_rf_states(self) -> List[List[Dict[str, Tensor]]]:
        """
        Initialize reshape-free TS_BLOCK states.

        Returns:
            List of states for each TS_BLOCK, where each state is a list of
            block states (one per time_stage block).
        """
        if self.rf_sequence_block is None:
            return []

        device = self.device
        dtype = next(self.model.parameters()).dtype

        all_states = []
        for rf_block in self.rf_sequence_block:
            block_states = rf_block.init_state(
                batch_size=1,
                freq_size=self.freq_size,
                device=device,
                dtype=dtype,
            )
            all_states.append(block_states)

        return all_states

    def _reset_rf_states(self) -> None:
        """Reset reshape-free TS_BLOCK states."""
        if self.use_reshape_free:
            self._rf_states = self._init_rf_states()

    def reset_state(self) -> None:
        """
        Reset all streaming state for a new audio stream.

        This resets:
        1. Audio input buffers
        2. Feature buffer for decoder lookahead
        3. Stateful convolution state buffers (for encoder/decoder)
        4. Reshape-free TS_BLOCK states (if enabled)
        """
        self._reset_buffers()

        # Reset reshape-free states if enabled
        if self.use_reshape_free:
            self._reset_rf_states()

        # Reset stateful convolution state (for encoder/decoder)
        from src.models.streaming.converters.conv_converter import (
            reset_streaming_state,
        )
        reset_streaming_state(self.model)

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration information."""
        return {
            **self._streaming_config,
            "chunk_size_frames": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "stft_lookahead_frames": self.stft_lookahead_frames,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "use_reshape_free": self.use_reshape_free,
            "freq_size": self.freq_size,
        }

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        stft_lookahead_frames: int = 1,
        use_reshape_free: bool = False,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "DuBLoNet":
        """
        Create DuBLoNet from a checkpoint directory.

        This automatically:
        1. Loads the model configuration and weights
        2. Converts convolutions to stateful versions
        3. Optionally converts TS_BLOCKs to reshape-free versions
        4. Reads encoder/decoder padding ratios from the model config for
           visibility/debugging (lookahead values are provided by the caller).

        Args:
            chkpt_dir: Path to checkpoint directory
            chkpt_file: Checkpoint file name (default: "best.th")
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames for encoder lookahead. Set to >0 if encoder
                has asymmetric padding (e.g., padding_ratio=(0.77, 0.23)).
                Set to 0 if encoder is fully causal (padding_ratio=(1.0, 0.0)).
            decoder_lookahead: Frames for decoder lookahead. Set to >0 if decoder
                has asymmetric padding. Set to 0 if decoder is fully causal.
            use_reshape_free: Convert TS_BLOCKs to reshape-free versions.
                Benefits:
                - Eliminates 16 reshape operations per inference
                - Unifies batch dimension to B=1 for all states
                - Reduces state count by ~48% (100 -> 52)
                - Recommended for NPU deployment
            device: Device to load model on
            verbose: Print loading information

        Returns:
            DuBLoNet instance

        Example configurations:
            - Decoder-only asymmetric: encoder_lookahead=0, decoder_lookahead=7
            - Both asymmetric: encoder_lookahead=7, decoder_lookahead=7
            - With reshape-free: use_reshape_free=True (recommended for mobile/NPU)
        """
        from src.models.streaming.utils import prepare_streaming_model

        if verbose:
            print(f"Loading DuBLoNet from: {chkpt_dir}")

        # Use unified model preparation pipeline
        model, metadata = prepare_streaming_model(
            chkpt_dir=chkpt_dir,
            chkpt_file=chkpt_file,
            use_stateful_conv=True,  # Always use stateful conv for buffered streaming
            use_reshape_free=use_reshape_free,
            device=device,
            verbose=verbose,
        )

        # Extract model args from metadata
        model_args = metadata["model_args"]
        model_params = model_args
        stateful_conv_count = metadata.get("stateful_conv_count", 0)
        rf_sequence_block = metadata.get("rf_sequence_block", None)
        rf_block_count = metadata.get("rf_block_count", 0)

        # Get padding ratios from model config
        enc_padding = getattr(model_params, 'encoder_padding_ratio', [1.0, 0.0])
        dec_padding = getattr(model_params, 'decoder_padding_ratio', [1.0, 0.0])

        if verbose:
            print(f"  Encoder padding ratio: {enc_padding}")
            print(f"  Decoder padding ratio: {dec_padding}")

        # Get STFT parameters
        hop_size = getattr(model_params, 'hop_size', 100)
        n_fft = getattr(model_params, 'n_fft', 400)
        win_size = getattr(model_params, 'win_size', 400)
        compress_factor = getattr(model_params, 'compress_factor', 0.3)

        # Calculate freq_size for state initialization
        freq_size = n_fft // 2 + 1  # Default to STFT frequency bins

        # Create instance
        instance = cls(
            model=model,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            stft_lookahead_frames=stft_lookahead_frames,
            hop_size=hop_size,
            n_fft=n_fft,
            win_size=win_size,
            compress_factor=compress_factor,
            rf_sequence_block=rf_sequence_block,
            freq_size=freq_size,
        )

        # Store config
        instance._streaming_config = {
            "chkpt_dir": chkpt_dir,
            "use_reshape_free": use_reshape_free,
            "encoder_padding_ratio": enc_padding,
            "decoder_padding_ratio": dec_padding,
            "stateful_conv_count": stateful_conv_count,
            "rf_block_count": rf_block_count,
            "model_class": "Backbone",
        }

        if verbose:
            print(f"  Chunk size: {chunk_size} frames")
            print(f"  Encoder lookahead: {encoder_lookahead} frames")
            print(f"  Decoder lookahead: {decoder_lookahead} frames")
            print(f"  Total latency: {instance.latency_ms:.1f}ms")
            if use_reshape_free:
                print(f"  Reshape-Free: {rf_block_count} TS_BLOCKs converted")

        return instance

    def _stft(self, audio: Tensor) -> Tensor:
        """Compute STFT and return complex spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        _, _, com = mag_pha_stft(
            audio,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=True
        )
        return com

    def _istft(self, mag: Tensor, pha: Tensor) -> Tensor:
        """Compute inverse STFT."""
        return mag_pha_istft(
            mag, pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=True
        )

    def _process_encoder(
        self,
        spectrogram: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Process spectrogram through Encoder + TS_BLOCK.

        This is the common encoder path for both immediate and buffered modes.
        When reshape-free is enabled, uses StatefulReshapeFreeTSBlock instead
        of the original TS_BLOCK with unified batch dimension.

        Args:
            spectrogram: Complex spectrogram [B, F, T, 2]

        Returns:
            Tuple of (mag, ts_out, valid_frames):
            - mag: Magnitude spectrogram [B, F, T]
            - ts_out: TS_BLOCK output [B, C, T, F]
            - valid_frames: Number of frames used for state update
        """
        from src.models.backbone import complex_to_mag_pha
        from src.models.streaming.utils import StateFramesContext

        _, _, T, _ = spectrogram.shape

        mag, pha = complex_to_mag_pha(spectrogram, stack_dim=-1)
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]

        # IMPORTANT:
        # STFT with center=True can produce extra frames in T depending on input windowing.
        # We only advance streaming state by `chunk_size` frames per step.
        # To prevent stateful layers from being updated by any extra frames,
        # we restrict state updates to the first `valid_frames` frames.
        valid_frames = min(T, self.chunk_size)

        with StateFramesContext(valid_frames):
            # Encoder (always use original model's encoder)
            encoded = self.model.dense_encoder(x)

            # TS_BLOCK: use reshape-free version if enabled
            if self.use_reshape_free and self.rf_sequence_block is not None:
                ts_out = self._process_rf_sequence_block(encoded)
            else:
                ts_out = self.model.sequence_block(encoded)  # [B, C, T, F]

        return mag, ts_out, valid_frames

    def _process_rf_sequence_block(self, x: Tensor) -> Tensor:
        """
        Process through reshape-free TS_BLOCKs with explicit state management.

        This uses StatefulReshapeFreeTSBlock which:
        - Eliminates reshape operations (no permute + contiguous)
        - Uses unified batch dimension (B=1 for all states)
        - freq_stage is stateless (no streaming state needed)

        Args:
            x: Encoder output [B, C, T, F]

        Returns:
            TS_BLOCK output [B, C, T, F]
        """
        if self.rf_sequence_block is None or self._rf_states is None:
            raise RuntimeError("Reshape-free sequence block not initialized")

        # Process through each TS_BLOCK with state
        for i, rf_block in enumerate(self.rf_sequence_block):
            x, new_states = rf_block(x, self._rf_states[i])
            self._rf_states[i] = new_states

        return x

    def _can_process_immediately(self, ts_out: Tensor) -> bool:
        """
        Determine if decoder can process immediately without buffering.

        Conditions for immediate processing:
        1. decoder_lookahead == 0
        2. encoder output frames >= chunk_size + stft_lookahead_frames

        This means "dec=0 AND enough frames available from encoder output for iSTFT".

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]

        Returns:
            True if immediate processing is possible, False otherwise.
        """
        if self.decoder_lookahead > 0:
            return False

        available_frames = ts_out.shape[2]
        needed_for_istft = self.chunk_size + self.stft_lookahead_frames

        return available_frames >= needed_for_istft

    def _process_decoder_immediate(
        self,
        ts_out: Tensor,
        mag: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process decoder immediately without feature buffer (decoder_lookahead=0).

        This path is used when:
        - decoder_lookahead == 0
        - encoder output has enough frames for iSTFT (>= chunk_size + stft_lookahead_frames)

        The encoder output's extra frames (from input_lookahead_frames) are used directly
        for iSTFT overlap-add, eliminating the need for feature buffer accumulation.

        IMPORTANT: StateFramesContext is used to limit state update to chunk_size,
        preventing lookahead frames from corrupting state for the next chunk.

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]
            mag: Magnitude spectrogram [B, F, T]

        Returns:
            Tuple of (est_mag, est_pha) ready for iSTFT
        """
        from src.models.streaming.utils import StateFramesContext

        frames_for_istft = self.chunk_size + self.stft_lookahead_frames

        # Use extended features from encoder output directly
        extended_features = ts_out[:, :, :frames_for_istft, :]
        extended_mag = mag[:, :, :frames_for_istft]

        # StateFramesContext limits state update to chunk_size
        # → lookahead frames (stft_lookahead_frames) do NOT corrupt state
        with StateFramesContext(self.chunk_size):
            mask = self.model.mask_decoder(extended_features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(extended_features).squeeze(1).transpose(1, 2)

        # Apply mask
        infer_type = getattr(self.model, 'infer_type', 'masking')
        if infer_type == 'masking':
            est_mag = extended_mag * mask
        else:
            est_mag = mask

        return est_mag, est_pha

    def _process_decoder_buffered(
        self,
        ts_out: Tensor,
        mag: Tensor,
        valid_frames: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Process decoder with feature buffering (decoder_lookahead > 0).

        This path accumulates encoded features in the feature buffer until
        enough frames are available for decoder processing with lookahead.

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]
            mag: Magnitude spectrogram [B, F, T]
            valid_frames: Number of frames used for encoder state update

        Returns:
            Tuple of (est_mag, est_pha) if output available, None otherwise
        """
        from src.models.streaming.utils import StateFramesContext

        # Step 1: Add to feature buffer
        # IMPORTANT: Only store valid_frames, not all T frames from STFT
        self.feature_buffer.append({
            'features': ts_out[:, :, :valid_frames, :],
            'mag': mag[:, :, :valid_frames],
            'frames': valid_frames,
        })
        self._buffered_frames += valid_frames

        # Step 2: Check if we have enough for decoder processing
        total_needed = self.chunk_size + self.decoder_lookahead + self.stft_lookahead_frames
        if self._buffered_frames < total_needed:
            logger.debug(f"Buffering: {self._buffered_frames}/{total_needed}")
            return None

        # Step 3: Gather features for decoder processing
        all_features = torch.cat([buf['features'] for buf in self.feature_buffer], dim=2)
        all_mag = torch.cat([buf['mag'] for buf in self.feature_buffer], dim=2)

        extended_features = all_features[:, :, :total_needed, :]
        extended_mag = all_mag[:, :, :total_needed]

        # Step 4: Process decoder with extended input
        with StateFramesContext(self.chunk_size):
            mask = self.model.mask_decoder(extended_features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(extended_features).squeeze(1).transpose(1, 2)

        # Apply mask
        infer_type = getattr(self.model, 'infer_type', 'masking')
        if infer_type == 'masking':
            est_mag = extended_mag * mask
        else:
            est_mag = mask

        # Step 5: Prepare output for iSTFT
        frames_for_istft = self.chunk_size + self.stft_lookahead_frames
        est_mag = est_mag[:, :, :frames_for_istft]
        est_pha = est_pha[:, :, :frames_for_istft]

        # Step 6: Update feature buffer
        frames_to_remove = self.chunk_size
        removed = 0

        while removed < frames_to_remove and self.feature_buffer:
            buf = self.feature_buffer[0]
            if buf['frames'] <= (frames_to_remove - removed):
                removed += buf['frames']
                self.feature_buffer.pop(0)
            else:
                keep_frames = buf['frames'] - (frames_to_remove - removed)
                buf['features'] = buf['features'][:, :, -keep_frames:, :]
                buf['mag'] = buf['mag'][:, :, -keep_frames:]
                buf['frames'] = keep_frames
                removed = frames_to_remove

        self._buffered_frames -= removed

        return est_mag, est_pha

    def process_spectrogram_buffered(
        self,
        spectrogram: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Process spectrogram using adaptive encoder-decoder approach.

        Processing flow:
        1. Encoder + TS_BLOCK: Process immediately (common path)
        2. Decoder path: Conditional based on decoder_lookahead
           - If decoder_lookahead=0 AND enough frames: Immediate processing
           - Otherwise: Buffer features and wait for lookahead

        Args:
            spectrogram: Complex spectrogram [B, F, T, 2] where T = chunk_size

        Returns:
            Tuple of (est_mag, est_pha) if output available, None otherwise
            Output time dimension is chunk_size + stft_lookahead_frames (for iSTFT)
        """
        with torch.no_grad():
            # Step 1: Encoder + TS_BLOCK (common path)
            mag, ts_out, valid_frames = self._process_encoder(spectrogram)

            # Step 2: Decoder path (conditional)
            if self._can_process_immediately(ts_out):
                # Immediate mode: Use encoder output directly for decoder
                return self._process_decoder_immediate(ts_out, mag)
            else:
                # Buffered mode: Accumulate features and wait for lookahead
                return self._process_decoder_buffered(ts_out, mag, valid_frames)

    def process_samples(self, samples: Tensor) -> Optional[Tensor]:
        """
        Process incoming audio samples and return enhanced output if available.

        Args:
            samples: Input audio samples [T] or [B, T] (B must be 1)

        Returns:
            Enhanced audio samples [T] if output available, None otherwise
        """
        if samples.dim() == 2:
            if samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            samples = samples.squeeze(0)

        samples = samples.to(self.device)

        # Ensure STFT context is on the same device (needed on first call after reset)
        if self._stft_context.device != samples.device:
            self._stft_context = self._stft_context.to(samples.device)

        # Add to buffer
        self.input_buffer = torch.cat([self.input_buffer.to(self.device), samples])

        # Check if we have enough samples for one chunk
        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        # Extract chunk for processing
        chunk_samples = self.input_buffer[:self.samples_per_chunk]

        # Prepend STFT context for proper center=True handling.
        # _stft_context is initialized with zeros (silence) and updated each step,
        # ensuring STFT frames align correctly across chunk boundaries.
        context_size = self.win_size // 2
        chunk_with_context = torch.cat([self._stft_context.to(chunk_samples.device), chunk_samples])
        spectrogram = self._stft(chunk_with_context)
        # Skip context frames (context_size / hop_size = 2 frames for win=400, hop=100)
        spectrogram = spectrogram[:, :, self._stft_context_frames:, :]

        # Save context for next chunk: the context_size samples immediately
        # before where the next chunk starts (= advance point).
        # When output_samples_per_chunk >= context_size, these are entirely
        # within the current buffer.  When output_samples_per_chunk < context_size
        # (e.g. chunk_size=1), we need to combine the tail of the previous
        # context with the current buffer's leading samples.
        advance = self.output_samples_per_chunk
        if advance >= context_size:
            self._stft_context = self.input_buffer[advance - context_size:advance].clone()
        else:
            need_from_prev = context_size - advance
            prev_part = self._stft_context[len(self._stft_context) - need_from_prev:]
            curr_part = self.input_buffer[:advance]
            self._stft_context = torch.cat([prev_part, curr_part]).clone()

        # Process through buffered model
        result = self.process_spectrogram_buffered(spectrogram)

        if result is None:
            # Still buffering for decoder lookahead
            # Advance input buffer to prepare for next chunk (by output size)
            self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]
            return None

        est_mag, est_pha = result

        # Compute iSTFT
        output_audio = self._istft(est_mag, est_pha).squeeze(0)
        valid_output = output_audio[:self.output_samples_per_chunk]

        # Update buffer
        self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]

        return valid_output

    def process_audio(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal in streaming fashion.

        Feeds audio through the per-chunk streaming pipeline (stateful conv +
        lookahead buffering + per-chunk iSTFT) and concatenates outputs.
        Silence padding is appended to flush all buffered data.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T'] where T' <= T
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        audio_length = len(audio)
        self.reset_state()
        outputs: List[Tensor] = []

        # Append silence to flush the pipeline
        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([
            audio,
            torch.zeros(flush_size, device=audio.device),
        ])

        # Feed in output-sized increments
        for i in range(0, len(padded), self.output_samples_per_chunk):
            chunk = padded[i:i + self.output_samples_per_chunk]
            if len(chunk) == 0:
                break
            result = self.process_samples(chunk)
            if result is not None and len(result) > 0:
                outputs.append(result)

        if not outputs:
            return torch.tensor([], device=audio.device)

        result = torch.cat(outputs)
        if len(result) > audio_length:
            result = result[:audio_length]
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.process_audio(x)

    def __repr__(self) -> str:
        config = self._streaming_config
        rf_info = ""
        if self.use_reshape_free:
            rf_count = config.get('rf_block_count', 0)
            rf_info = f"  use_reshape_free=True ({rf_count} blocks),\n"
        return (
            f"{self.__class__.__name__}(\n"
            f"{rf_info}"
            f"  chunk_size={self.chunk_size},\n"
            f"  encoder_lookahead={self.encoder_lookahead},\n"
            f"  decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_lookahead={self.total_lookahead},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f")"
        )
