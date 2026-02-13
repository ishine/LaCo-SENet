"""
Common ONNX export verification utilities.

Shared multi-step verification logic used by both stateful_core and
stateful_core_rf verify functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor


def verify_stateful_onnx_multistep(
    onnx_path: str,
    core: torch.nn.Module,
    init_states_fn: Callable[..., List[Tensor]],
    get_state_names_fn: Callable[[], List[str]],
    num_non_state_outputs: int = 2,
    batch_size: int = 1,
    time_frames: int = 64,
    freq_size: int = 129,
    num_steps: int = 5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Verify stateful ONNX export against PyTorch model over multiple steps.

    This is the shared implementation for both regular stateful and
    reshape-free stateful ONNX verification.

    Args:
        onnx_path: Path to ONNX file
        core: Original PyTorch model (must accept (mag, pha, *states))
        init_states_fn: Callable that returns initial state tensors.
            Called as init_states_fn(batch_size, freq_size, time_frames, device, dtype).
        get_state_names_fn: Callable that returns state names for ONNX input mapping.
        num_non_state_outputs: Number of non-state outputs (2 for mag/pha,
            3 if complex phase output is used).
        batch_size: Batch size for test
        time_frames: Number of time frames per step
        freq_size: Frequency dimension
        num_steps: Number of steps to run
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: If True, print summary at the end

    Returns:
        Dict with verification results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    import numpy as np

    core.eval()
    device = next(core.parameters()).device

    # Initialize ONNX session
    sess = ort.InferenceSession(onnx_path)

    # Initialize states
    pt_states = init_states_fn(batch_size, freq_size, time_frames, device, torch.float32)
    ort_states = [s.cpu().numpy() for s in pt_states]

    state_names = get_state_names_fn()

    results: Dict[str, Any] = {
        "steps": [],
        "mask_max_diffs": [],
        "pha_max_diffs": [],
        "state_max_diffs": [],
    }

    for step in range(num_steps):
        # Create random input for this step
        mag = torch.randn(batch_size, freq_size, time_frames, device=device)
        pha = torch.randn(batch_size, freq_size, time_frames, device=device)

        # PyTorch inference
        with torch.no_grad():
            pt_outputs = core(mag, pha, *pt_states)
            pt_mask = pt_outputs[0].cpu().numpy()
            pt_pha = pt_outputs[1].cpu().numpy()
            pt_next_states = [s.cpu().numpy() for s in pt_outputs[num_non_state_outputs:]]

        # ONNX Runtime inference
        ort_inputs = {
            "mag": mag.cpu().numpy(),
            "pha": pha.cpu().numpy(),
        }
        for i, sname in enumerate(state_names):
            ort_inputs[sname] = ort_states[i]

        ort_outputs = sess.run(None, ort_inputs)
        ort_mask = ort_outputs[0]
        ort_pha = ort_outputs[1]
        ort_next_states = ort_outputs[num_non_state_outputs:]

        # Compare outputs
        mask_diff = np.abs(pt_mask - ort_mask).max()
        pha_diff = np.abs(pt_pha - ort_pha).max()
        state_diffs = [
            np.abs(pt_next_states[i] - ort_next_states[i]).max()
            for i in range(len(pt_next_states))
        ]

        results["steps"].append(step)
        results["mask_max_diffs"].append(float(mask_diff))
        results["pha_max_diffs"].append(float(pha_diff))
        results["state_max_diffs"].append([float(d) for d in state_diffs])

        # Update states for next step
        pt_states = list(pt_outputs[num_non_state_outputs:])
        ort_states = list(ort_next_states)

    # Aggregate results
    all_mask_match = all(d <= atol for d in results["mask_max_diffs"])
    all_pha_match = all(d <= atol for d in results["pha_max_diffs"])

    results["all_match"] = all_mask_match and all_pha_match
    results["max_mask_diff"] = max(results["mask_max_diffs"])
    results["max_pha_diff"] = max(results["pha_max_diffs"])

    if verbose:
        status = "PASS" if results["all_match"] else "FAIL"
        print(f"Verification ({num_steps} steps): {status}")
        print(f"  Max mask diff: {results['max_mask_diff']:.6e}")
        print(f"  Max pha diff:  {results['max_pha_diff']:.6e}")

    return results
