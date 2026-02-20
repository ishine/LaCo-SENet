"""Receptive field calculator for Backbone.

Computes the theoretical (convolutional) receptive field size
from model config parameters, broken down by component.

Usage:
    from src.receptive_field import compute_receptive_field

    rf = compute_receptive_field(cfg.model.param)
    rf.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union


@dataclass
class ComponentRF:
    """Receptive field info for a single component."""
    name: str
    rf_frames: int
    detail: str = ""


@dataclass
class ReceptiveFieldResult:
    """Full receptive field analysis result."""
    components: List[ComponentRF] = field(default_factory=list)
    total_rf_frames: int = 0
    hop_len: int = 100
    win_len: int = 400
    sampling_rate: int = 16000

    @property
    def total_rf_samples(self) -> int:
        return (self.total_rf_frames - 1) * self.hop_len + self.win_len

    @property
    def total_rf_ms(self) -> float:
        return self.total_rf_samples / self.sampling_rate * 1000

    @property
    def one_sided_rf_frames(self) -> float:
        return (self.total_rf_frames - 1) / 2

    @property
    def one_sided_rf_ms(self) -> float:
        return self.one_sided_rf_frames * self.hop_len / self.sampling_rate * 1000

    def summary(self) -> str:
        lines = []
        lines.append("=" * 62)
        lines.append("Backbone Receptive Field Analysis")
        lines.append("=" * 62)

        lines.append("")
        lines.append("Component Breakdown:")
        lines.append(f"  {'Component':<30} {'RF (frames)':>12}")
        lines.append(f"  {'-'*30} {'-'*12}")
        for c in self.components:
            lines.append(f"  {c.name:<30} {c.rf_frames:>12}")
            if c.detail:
                lines.append(f"    {c.detail}")
        lines.append(f"  {'-'*30} {'-'*12}")
        lines.append(f"  {'TOTAL':<30} {self.total_rf_frames:>12}")

        lines.append("")
        lines.append("Time Domain Conversion:")
        lines.append(f"  RF (samples)       : {self.total_rf_samples}")
        lines.append(f"  RF (ms)            : {self.total_rf_ms:.1f}")
        lines.append(f"  RF (sec)           : {self.total_rf_ms / 1000:.4f}")
        lines.append(f"  One-sided (frames) : {self.one_sided_rf_frames:.1f}")
        lines.append(f"  One-sided (ms)     : {self.one_sided_rf_ms:.1f}")

        lines.append("")
        lines.append("Global Operations (make actual RF = full sequence):")
        lines.append("  - AdaptiveAvgPool1d in SCA (TS_BLOCK)")
        lines.append("=" * 62)
        return "\n".join(lines)

    def print(self) -> None:
        print(self.summary())


def _ds_ddb_rf(kernel_size: int, depth: int) -> int:
    """RF of DS_DDB (Dense Dilated Depthwise Block).

    Each layer has dilation = 2^i, so effective kernel = 2^i * (k-1) + 1.
    Dense connections make RF = 1 + sum(eff_k_i - 1) for i in 0..depth-1.
    """
    return 1 + sum(2**i * (kernel_size - 1) for i in range(depth))


def _encoder_rf(dense_depth: int, kernel_size: int = 3) -> int:
    """DenseEncoder RF on the time axis.

    Structure: Conv2d(1,1) -> DS_DDB -> Conv2d(1,3) stride=(1,2)
    Only DS_DDB contributes to time-axis RF (other kernels are 1 in time).
    """
    return _ds_ddb_rf(kernel_size, dense_depth)


def _cab_rf(dw_kernel_size: int) -> int:
    """Channel_Attention_Block local conv RF (excluding SCA global pooling)."""
    return dw_kernel_size


def _gpkffn_rf(kernel_list: Sequence[int]) -> int:
    """Group_Prime_Kernel_FFN RF.

    Parallel branches with different kernels, mixed by 1x1 conv.
    RF = max(kernel_list).
    """
    return max(kernel_list)


def _ts_block_time_rf(
    time_block_num: int,
    time_dw_kernel_size: int,
    time_block_kernel: Sequence[int],
) -> int:
    """One TS_BLOCK's time-axis RF.

    Each iteration = CAB + GPKFFN (sequential).
    Repeated time_block_num times.
    Freq stage does not affect time RF.
    """
    cab_rf = _cab_rf(time_dw_kernel_size)
    gpkffn_rf = _gpkffn_rf(time_block_kernel)
    one_iter_rf = (cab_rf - 1) + (gpkffn_rf - 1) + 1
    return time_block_num * (one_iter_rf - 1) + 1


def _decoder_rf(dense_depth: int, kernel_size: int = 3) -> int:
    """MaskDecoder / PhaseDecoder RF on the time axis.

    Structure: DS_DDB -> ConvTranspose2d(1,3) -> Conv2d(1,1) ...
    Only DS_DDB contributes to time-axis RF.
    """
    return _ds_ddb_rf(kernel_size, dense_depth)


def compute_receptive_field(
    param: Union[Dict, object],
    hop_len: int | None = None,
    win_len: int | None = None,
    sampling_rate: int = 16000,
) -> ReceptiveFieldResult:
    """Compute Backbone receptive field from model parameters.

    Args:
        param: Model parameter dict or config object with attributes:
            - dense_depth (int): Depth of DS_DDB (default: 4)
            - num_tsblock (int): Number of TS_BLOCKs (default: 4)
            - time_block_num (int): Iterations per TS_BLOCK time stage (default: 2)
            - time_dw_kernel_size (int): CAB depthwise kernel size (default: 3)
            - time_block_kernel (list[int]): GPKFFN kernel list (default: [3,11,23,31])
            - hop_len (int): STFT hop length (default: 100)
            - win_len (int): STFT window length (default: 400)
        hop_len: Override hop_len from param.
        win_len: Override win_len from param.
        sampling_rate: Audio sampling rate in Hz.

    Returns:
        ReceptiveFieldResult with component breakdown and totals.
    """
    def _get(key, default=None):
        if isinstance(param, dict):
            return param.get(key, default)
        return getattr(param, key, default)

    dense_depth = _get("dense_depth", 4)
    num_tsblock = _get("num_tsblock", 4)
    time_block_num = _get("time_block_num", 2)
    time_dw_kernel_size = _get("time_dw_kernel_size", 3)
    time_block_kernel = list(_get("time_block_kernel", [3, 11, 23, 31]))
    _hop = hop_len or _get("hop_len", 100)
    _win = win_len or _get("win_len", 400)

    # Component RFs
    enc_rf = _encoder_rf(dense_depth)
    one_ts_rf = _ts_block_time_rf(time_block_num, time_dw_kernel_size, time_block_kernel)
    all_ts_rf = num_tsblock * (one_ts_rf - 1) + 1
    dec_rf = _decoder_rf(dense_depth)

    total_rf = (enc_rf - 1) + (all_ts_rf - 1) + (dec_rf - 1) + 1

    # Build component list
    cab_rf = _cab_rf(time_dw_kernel_size)
    gpkffn_rf = _gpkffn_rf(time_block_kernel)
    one_iter_rf = (cab_rf - 1) + (gpkffn_rf - 1) + 1

    components = [
        ComponentRF(
            "DenseEncoder (DS_DDB)",
            enc_rf,
            f"depth={dense_depth}, dil=[{','.join(str(2**i) for i in range(dense_depth))}]",
        ),
        ComponentRF(
            f"TS_BLOCK x{num_tsblock}",
            all_ts_rf,
            f"per block={one_ts_rf} "
            f"(CAB(k={time_dw_kernel_size})={cab_rf} + "
            f"GPKFFN(max={max(time_block_kernel)})={gpkffn_rf} "
            f"-> iter={one_iter_rf}, x{time_block_num}iter={one_ts_rf})",
        ),
        ComponentRF(
            "Decoder (DS_DDB)",
            dec_rf,
            f"depth={dense_depth}, dil=[{','.join(str(2**i) for i in range(dense_depth))}]",
        ),
    ]

    return ReceptiveFieldResult(
        components=components,
        total_rf_frames=total_rf,
        hop_len=_hop,
        win_len=_win,
        sampling_rate=sampling_rate,
    )


def rf_to_segment(
    param: Union[Dict, object],
    hop_len: int | None = None,
    win_len: int | None = None,
    sampling_rate: int = 16000,
    center: bool = True,
) -> int:
    """Compute the minimum training segment size (in samples) from model RF.

    Returns RF in samples, aligned up to the nearest multiple of hop_len.

    When center=False, adds n_fft samples to compensate for the lack of
    center padding, ensuring the same number of STFT frames as center=True.
    """
    rf = compute_receptive_field(param, hop_len, win_len, sampling_rate)
    hop = rf.hop_len
    samples = rf.total_rf_samples
    if not center:
        n_fft = rf.win_len  # n_fft == win_len in this codebase
        samples += n_fft
    # Align up to hop_len multiple
    aligned = ((samples + hop - 1) // hop) * hop
    return aligned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backbone Receptive Field Calculator")
    parser.add_argument("--dense-depth", type=int, default=4)
    parser.add_argument("--num-tsblock", type=int, default=4)
    parser.add_argument("--time-block-num", type=int, default=2)
    parser.add_argument("--time-dw-kernel-size", type=int, default=3)
    parser.add_argument("--time-block-kernel", type=int, nargs="+", default=[3, 11, 23, 31])
    parser.add_argument("--hop-len", type=int, default=100)
    parser.add_argument("--win-len", type=int, default=400)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()

    result = compute_receptive_field(
        param={
            "dense_depth": args.dense_depth,
            "num_tsblock": args.num_tsblock,
            "time_block_num": args.time_block_num,
            "time_dw_kernel_size": args.time_dw_kernel_size,
            "time_block_kernel": args.time_block_kernel,
            "hop_len": args.hop_len,
            "win_len": args.win_len,
        },
        sampling_rate=args.sampling_rate,
    )
    result.print()
