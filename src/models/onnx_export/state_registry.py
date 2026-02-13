"""
State Registry for ONNX Export.

This module provides utilities for collecting and managing state tensors
from a model for explicit state I/O in ONNX export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class StateInfo:
    """Information about a single state tensor."""
    name: str  # Unique name for ONNX I/O
    module_path: str  # Path to the module in the model hierarchy
    shape: Tuple[int, ...]  # Expected shape (with batch dim)
    dtype: torch.dtype
    init_fn: Any  # Callable to initialize the state


class StateRegistry:
    """
    Registry for managing state tensors in ONNX export.

    This class:
    1. Collects all stateful layers from a model
    2. Assigns unique names to state tensors
    3. Provides initialization and indexing utilities
    """

    def __init__(self):
        self._states: List[StateInfo] = []
        self._name_to_idx: Dict[str, int] = {}

    def register(
        self,
        name: str,
        module_path: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        init_fn: Any = None,
    ) -> int:
        """
        Register a state tensor.

        Args:
            name: Unique name for the state
            module_path: Path to the module (e.g., "dense_encoder.dense_block.0.conv")
            shape: Expected shape with batch dimension
            dtype: Data type
            init_fn: Function to initialize the state (receives device, dtype)

        Returns:
            Index of the registered state
        """
        if name in self._name_to_idx:
            raise ValueError(f"State '{name}' already registered")

        idx = len(self._states)
        info = StateInfo(
            name=name,
            module_path=module_path,
            shape=shape,
            dtype=dtype,
            init_fn=init_fn or (lambda dev, dt: torch.zeros(shape, device=dev, dtype=dt)),
        )
        self._states.append(info)
        self._name_to_idx[name] = idx
        return idx

    def get_by_name(self, name: str) -> StateInfo:
        """Get state info by name."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown state: {name}")
        return self._states[idx]

    def get_by_index(self, idx: int) -> StateInfo:
        """Get state info by index."""
        return self._states[idx]

    @property
    def num_states(self) -> int:
        """Number of registered states."""
        return len(self._states)

    @property
    def state_names(self) -> List[str]:
        """List of state names in order."""
        return [s.name for s in self._states]

    def init_all_states(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """
        Initialize all states.

        Args:
            batch_size: Batch size (replaces first dim if > 1)
            device: Device for state tensors
            dtype: Data type for state tensors

        Returns:
            List of initialized state tensors in registration order
        """
        states = []
        for info in self._states:
            # Adjust batch dimension if needed
            shape = (batch_size,) + info.shape[1:]
            t = torch.zeros(shape, device=device, dtype=dtype or info.dtype)
            states.append(t)
        return states

    def to_dict(self, states: List[Tensor]) -> Dict[str, Tensor]:
        """Convert state list to named dict."""
        if len(states) != len(self._states):
            raise ValueError(f"Expected {len(self._states)} states, got {len(states)}")
        return {info.name: states[i] for i, info in enumerate(self._states)}

    def from_dict(self, state_dict: Dict[str, Tensor]) -> List[Tensor]:
        """Convert named dict to state list."""
        states = []
        for info in self._states:
            if info.name not in state_dict:
                raise KeyError(f"Missing state: {info.name}")
            states.append(state_dict[info.name])
        return states

    def summary(self) -> str:
        """Return a summary of all registered states."""
        lines = [f"StateRegistry: {self.num_states} states"]
        total_elements = 0
        for i, info in enumerate(self._states):
            elements = 1
            for d in info.shape:
                elements *= d
            total_elements += elements
            lines.append(f"  [{i}] {info.name}: {info.shape} ({info.dtype})")
        lines.append(f"  Total elements: {total_elements:,}")
        return "\n".join(lines)


def collect_states_from_model(
    model: torch.nn.Module,
    batch_size: int = 1,
    freq_size: int = 129,  # n_fft // 2 + 1 for n_fft=256
) -> Tuple[StateRegistry, List[Tensor]]:
    """
    Collect all state tensors from a model with functional layers.

    This function walks the model tree and finds all modules with
    `init_state` methods, registering their states.

    Args:
        model: Model with functional stateful layers
        batch_size: Batch size for state initialization
        freq_size: Frequency size for 2D conv states

    Returns:
        Tuple of (registry, initial_states)
    """
    from src.models.onnx_export.layers.functional_stateful import (
        FunctionalStatefulCausalConv2d,
        FunctionalStatefulConv1d,
        FunctionalStatefulConv2d,
    )

    registry = StateRegistry()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for name, module in model.named_modules():
        if isinstance(module, FunctionalStatefulConv1d):
            state_name = f"state_{name.replace('.', '_')}"
            state = module.init_state(batch_size, device, dtype)
            registry.register(
                name=state_name,
                module_path=name,
                shape=tuple(state.shape),
                dtype=dtype,
            )

        elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
            state_name = f"state_{name.replace('.', '_')}"
            state = module.init_state(batch_size, freq_size, device, dtype)
            registry.register(
                name=state_name,
                module_path=name,
                shape=tuple(state.shape),
                dtype=dtype,
            )

    initial_states = registry.init_all_states(batch_size, device, dtype)
    return registry, initial_states


__all__ = [
    "StateInfo",
    "StateRegistry",
    "collect_states_from_model",
]
