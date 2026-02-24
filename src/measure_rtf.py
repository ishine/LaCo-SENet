# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
"""Measure Real-Time Factor (RTF) for streaming Backbone models.

Feeds dummy audio through the DuBLoNet streaming pipeline chunk by chunk
and reports RTF (wall-clock / audio duration).

Default arguments are tuned for single-thread CPU RTF measurement:
    --num_threads  1      Single-core baseline
    --warmup       3      Cache/JIT stabilization
    --repeats      3      Statistical reliability (mean ± std)
    --chunk_size   1      Per-frame granularity (worst-case RTF)
    --duration     2.0    2-second dummy utterance

Reshape-free and BN folding are always enabled.

Usage:
    python -m src.measure_rtf \\
        --chkpt_dir results/experiments/M1_6.25ms/s7 \\
        --chkpt_file model_130000.th
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import torch

from src.utils import load_model, load_checkpoint


@dataclass
class RTFResult:
    mean: float
    std: float
    per_repeat: List[float]


def compute_lookahead_from_config(model_config) -> Tuple[int, int]:
    """Compute encoder/decoder lookahead frames from model config.

    Mirrors the DS_DDB padding logic in backbone.py: for each dilated layer,
    the right (future) padding determines required lookahead frames.

    Args:
        model_config: OmegaConf model config node

    Returns:
        (encoder_lookahead, decoder_lookahead) in frames
    """
    depth = getattr(model_config, 'dense_depth', 4)
    enc_ratio = list(getattr(model_config, 'encoder_padding_ratio', [0.5, 0.5]))
    dec_ratio = list(getattr(model_config, 'decoder_padding_ratio', [0.5, 0.5]))

    def _calc_lookahead(left_ratio: float, depth: int) -> int:
        lookahead = 0
        for i in range(depth):
            dil = 2 ** i
            # get_padding_2d((3,3), (dil,1)) -> time padding_one_side = dil
            time_padding_total = dil * 2
            time_padding_left = round(time_padding_total * left_ratio)
            time_padding_right = time_padding_total - time_padding_left
            lookahead += time_padding_right
        return lookahead

    enc_la = _calc_lookahead(enc_ratio[0], depth)
    dec_la = _calc_lookahead(dec_ratio[0], depth)
    return enc_la, dec_la


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def generate_dummy_audio(
    duration_sec: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Generate a single dummy audio tensor for RTF measurement."""
    samples = int(duration_sec * sample_rate)
    return torch.randn(samples)


def measure_streaming_rtf(
    streaming_model,
    audio: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Tuple[RTFResult, RTFResult]:
    """Measure RTF in streaming mode.

    Feeds a single utterance through process_samples() chunk by chunk,
    including the full STFT/model/iSTFT pipeline.

    Returns:
        (steady_state_result, total_result) — steady-state excludes flush
        iterations that drain the lookahead buffer.
    """
    osp = streaming_model.output_samples_per_chunk
    total_audio_sec = len(audio) / 16000.0
    audio_iters = math.ceil(len(audio) / osp)

    flush = streaming_model.samples_per_chunk * (streaming_model.total_lookahead + 2)
    padded = torch.cat([audio, torch.zeros(flush)])

    per_repeat_ss = []
    per_repeat_total = []
    for r in range(warmup + repeats):
        streaming_model.reset_state()
        t0 = time.perf_counter()
        t_audio = None
        iter_idx = 0

        with torch.inference_mode():
            for i in range(0, len(padded), osp):
                chunk = padded[i:i + osp]
                if len(chunk) == 0:
                    break
                streaming_model.process_samples(chunk)
                iter_idx += 1
                if iter_idx == audio_iters:
                    t_audio = time.perf_counter() - t0

        elapsed = time.perf_counter() - t0
        if t_audio is None:
            t_audio = elapsed
        if r >= warmup:
            per_repeat_ss.append(t_audio / total_audio_sec)
            per_repeat_total.append(elapsed / total_audio_sec)

    ss = RTFResult(
        mean=float(np.mean(per_repeat_ss)),
        std=float(np.std(per_repeat_ss)),
        per_repeat=per_repeat_ss,
    )
    total = RTFResult(
        mean=float(np.mean(per_repeat_total)),
        std=float(np.std(per_repeat_total)),
        per_repeat=per_repeat_total,
    )
    return ss, total



def get_hardware_info() -> str:
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description="Measure RTF for Backbone model")
    parser.add_argument("--chkpt_dir", type=str, required=True)
    parser.add_argument("--chkpt_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration of dummy utterance in seconds")
    parser.add_argument("--use_onnx", action="store_true",
                        help="Use ONNX Runtime backend instead of PyTorch")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    # Set CPU threads before any tensor ops
    torch.set_num_threads(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    from omegaconf import OmegaConf

    conf = OmegaConf.load(os.path.join(args.chkpt_dir, '.hydra', 'config.yaml'))

    # Compute lookahead from config
    enc_la, dec_la = compute_lookahead_from_config(conf.model)

    # Generate dummy audio
    test_audio = generate_dummy_audio(args.duration)
    total_audio_sec = len(test_audio) / 16000.0

    # Load model for param count
    device = "cpu"
    model = load_model(conf.model, device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, device)
    num_params = count_parameters(model)

    # Latency info
    hop_size = conf.model.get("hop_len", conf.model.fft_len // 4)
    win_size = conf.model.get("win_len", conf.model.fft_len)
    stft_center = conf.model.get("stft_center", True)
    stft_delay = win_size // 2 if stft_center else 0
    latency_samples = (enc_la + dec_la) * hop_size + stft_delay
    latency_ms = latency_samples / 16000.0 * 1000

    hardware = get_hardware_info()

    backend = "onnx" if args.use_onnx else "pytorch"

    print(f"\n=== RTF Measurement Results ===")
    print(f"Hardware: {hardware} ({args.num_threads} thread{'s' if args.num_threads > 1 else ''})")
    print(f"Backend: {backend}")
    if args.use_onnx:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__} (providers: {ort.get_available_providers()})")
    print(f"Streaming mode (chunk_size={args.chunk_size})")
    print(f"Test audio: {args.duration:.1f}s dummy utterance")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print()

    # Streaming RTF
    print("Creating streaming model...")
    if args.use_onnx:
        from src.models.onnx_export import ONNXDuBLoNet

        streaming_model = ONNXDuBLoNet.from_checkpoint(
            chkpt_dir=args.chkpt_dir,
            chkpt_file=args.chkpt_file,
            chunk_size=args.chunk_size,
            encoder_lookahead=enc_la,
            decoder_lookahead=dec_la,
            use_reshape_free=True,
            verbose=True,
        )
    else:
        from src.models.streaming.dublonet import DuBLoNet

        streaming_model = DuBLoNet.from_checkpoint(
            chkpt_dir=args.chkpt_dir,
            chkpt_file=args.chkpt_file,
            chunk_size=args.chunk_size,
            encoder_lookahead=enc_la,
            decoder_lookahead=dec_la,
            use_reshape_free=True,
            fold_bn=True,
            device=device,
            verbose=True,
        )

    print("Measuring streaming RTF...")
    ss_result, total_result = measure_streaming_rtf(
        streaming_model=streaming_model,
        audio=test_audio,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    chunk_audio_sec = args.chunk_size * hop_size / 16000.0
    ss_per_chunk_ms = ss_result.mean * chunk_audio_sec * 1000
    print(f"Streaming RTF:  {ss_result.mean:.4f} ± {ss_result.std:.4f}  ({ss_per_chunk_ms:.3f}ms per chunk)  [steady-state]")
    if abs(total_result.mean - ss_result.mean) > 0.01:
        print(f"  (with flush: {total_result.mean:.4f} ± {total_result.std:.4f})")

    print()
    print(f"Model: {num_params / 1e6:.2f}M params")
    print(f"Latency: {latency_ms:.1f}ms (enc_la={enc_la}, dec_la={dec_la})")

    # JSON output
    if args.output_json:
        output = {
            "hardware": hardware,
            "num_threads": args.num_threads,
            "device": device,
            "backend": backend,
            "streaming_rtf": asdict(ss_result),
            "streaming_rtf_total": asdict(total_result),
            "model_info": {
                "params": num_params,
                "latency_ms": latency_ms,
                "enc_lookahead": enc_la,
                "dec_lookahead": dec_la,
                "encoder_padding_ratio": list(conf.model.encoder_padding_ratio),
                "decoder_padding_ratio": list(conf.model.decoder_padding_ratio),
            },
            "test_info": {
                "duration_sec": args.duration,
            },
            "config": {
                "chunk_size": args.chunk_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "use_reshape_free": True,
                "fold_bn": True,
                "use_onnx": args.use_onnx,
            },
        }
        if args.use_onnx:
            import onnxruntime as ort
            output["onnx_info"] = {
                "onnxruntime_version": ort.__version__,
                "providers": ort.get_available_providers(),
            }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
