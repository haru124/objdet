"""
utils/common.py

Miscellaneous utilities shared across modules.
"""

import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Also enables deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes cuDNN operations deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "cuda") -> torch.device:
    """
    Return a torch.device, falling back to CPU if CUDA is unavailable.

    Args:
        requested: "cuda" or "cpu"
    """
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU.")
    return device


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_config_dict(cfg) -> dict:
    """
    Flatten a TrainingPipelineConfig into a dict suitable for MLflow / TB
    hparams logging.

    Each sub-config field is prefixed, e.g. "training.learning_rate".
    """
    from dataclasses import fields, is_dataclass

    result = {}

    def _flatten(obj, prefix=""):
        if is_dataclass(obj):
            for f in fields(obj):
                _flatten(getattr(obj, f.name), prefix=f"{prefix}{f.name}.")
        else:
            key = prefix.rstrip(".")
            # Only log serialisable primitives
            if isinstance(obj, (int, float, str, bool)):
                result[key] = obj

    _flatten(cfg)
    return result