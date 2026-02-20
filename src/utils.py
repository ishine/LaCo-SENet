import os
import time
import torch
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Union, Any
from joblib import Parallel, delayed
from contextlib import contextmanager
import atexit
from pesq import pesq

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss

# Reusable joblib Parallel pool (loky backend)
_JOBLIB_PARALLEL = None
_JOBLIB_WORKERS = None

def _get_joblib_parallel(workers: int):
    """Return a reusable joblib Parallel instance; recreate if worker size changed."""
    global _JOBLIB_PARALLEL, _JOBLIB_WORKERS
    if _JOBLIB_PARALLEL is None or _JOBLIB_WORKERS != workers:
        # Terminate existing pool if present (best-effort; uses private API)
        if _JOBLIB_PARALLEL is not None:
            try:
                _JOBLIB_PARALLEL._terminate_pool()
            except Exception:
                pass
        _JOBLIB_PARALLEL = Parallel(n_jobs=workers, backend="loky", prefer="processes")
        _JOBLIB_WORKERS = workers
    return _JOBLIB_PARALLEL

def batch_pesq(clean, noisy, workers=8, normalize=True):
    # Reuse a single loky process pool to avoid frequent creation/cleanup cycles
    parallel = _get_joblib_parallel(workers)
    pesq_score = parallel(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    if normalize:
        pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext



class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos
    
    def append(self, **infos):
        self._infos.update(**infos)
        
    def _append(self, info):
        self._infos.update(info)

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


# ============================================================================
# Model and Checkpoint Utilities
# ============================================================================

def load_model(model_params: Dict[str, Any], device: str = 'cuda'):
    """
    Create Backbone model instance.

    Args:
        model_params: Model parameters dictionary
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Backbone instance loaded on specified device

    Example:
        >>> model = load_model(params, "cuda")
    """
    from src.models.backbone import Backbone
    # Filter out pipeline-level keys that are not Backbone constructor params
    _non_backbone_keys = {"stft_center"}
    backbone_params = {k: v for k, v in model_params.items() if k not in _non_backbone_keys}
    model = Backbone(**backbone_params)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, chkpt_dir: str, chkpt_file: str, device: str = 'cuda'):
    """
    Load model checkpoint.

    Args:
        model: Model instance to load checkpoint into
        chkpt_dir: Directory containing checkpoint file
        chkpt_file: Checkpoint filename (e.g., "best.th", "checkpoint.th")
        device: Device for loading checkpoint

    Returns:
        Model with loaded checkpoint

    Example:
        >>> model = load_checkpoint(model, "outputs/exp", "best.th", "cuda")
    """
    chkpt_path = os.path.join(chkpt_dir, chkpt_file)
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chkpt['model'])
    return model


class ConfigDict:
    """
    Convert nested dictionary to object with attribute access.

    Useful for converting checkpoint args to object-like structure.

    Example:
        >>> config = ConfigDict({'model': {'name': 'test', 'params': {'lr': 0.01}}})
        >>> config.model.name  # 'test'
        >>> config.model.params.lr  # 0.01
        >>> config.to_dict()  # {'model': {'name': 'test', 'params': {'lr': 0.01}}}
    """

    def __init__(self, d: Dict[str, Any]):
        for key, value in d.items():
            setattr(self, key, value if not isinstance(value, dict) else ConfigDict(value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict back to regular dict for ** unpacking"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_model_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model configuration from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., "outputs/exp/best.th")

    Returns:
        Dictionary of model parameters (e.g., {'dense_channel': 64, ...})

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain 'args' or 'args.model'

    Example:
        >>> config = load_model_config_from_checkpoint("outputs/backbone_exp/best.th")
        >>> print(config['dense_channel'])  # 64
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint (CPU is sufficient for config extraction)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'args' field: {checkpoint_path}")

    args = checkpoint['args']

    if not hasattr(args, 'model'):
        raise KeyError(f"Checkpoint args does not contain 'model' field: {checkpoint_path}")

    model_config = args.model

    return dict(model_config)


def parse_file_list(directory: str, list_file: str) -> List[str]:
    """
    Parse file list and return full paths.

    Args:
        directory: Base directory path
        list_file: Path to file containing list of filenames (one per line)

    Returns:
        List of full file paths

    Example:
        >>> noise_files = parse_file_list("/data/noise", "dataset/noise_train.txt")
    """
    with open(list_file, 'r') as f:
        return [os.path.join(directory, line.strip()) for line in f]


def get_stft_args_from_config(model_args) -> Dict[str, Any]:
    """
    Extract STFT arguments from model configuration.

    Args:
        model_args: Model configuration object (args.model)

    Returns:
        Dictionary of STFT arguments

    Example:
        >>> stft_args = get_stft_args_from_config(config.model)
        >>> # Returns: {"n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3}
    """
    fft_len = model_args.fft_len
    return {
        "n_fft": fft_len,
        "hop_size": model_args.get("hop_len", fft_len // 4),
        "win_size": model_args.get("win_len", fft_len),
        "compress_factor": model_args.get("compress_factor", 1.0),
        "center": model_args.get("stft_center", True),
    }
