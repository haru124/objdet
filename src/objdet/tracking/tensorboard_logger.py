"""
tracking/tensorboard_logger.py

Thin wrapper around torch.utils.tensorboard.SummaryWriter.

Keeping logging behind a wrapper means:
  - The rest of the code doesn't need to guard against tb being unavailable.
  - It's easy to swap implementations (e.g. W&B) without touching training code.
"""

from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TensorBoardLogger:
    """
    Wraps SummaryWriter.

    Args:
        log_dir:         Root directory for TensorBoard event files.
        experiment_name: Sub-folder name; helps separate multiple runs.
    """

    def __init__(self, log_dir: str | Path, experiment_name: str = "run"):
        run_dir = Path(log_dir) / experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if _TB_AVAILABLE:
            self._writer = SummaryWriter(log_dir=str(run_dir))
            print(f"[TensorBoard] Logging to: {run_dir}")
        else:
            self._writer = None
            print("[TensorBoard] tensorboard not installed — logging disabled.")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int):
        if self._writer:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag: str, value_dict: dict[str, float], step: int):
        """Log multiple scalars under the same tag group."""
        if self._writer:
            self._writer.add_scalars(tag, value_dict, global_step=step)

    def log_image(self, tag: str, image_tensor, step: int):
        """Log a CHW image tensor (values in [0,1])."""
        if self._writer:
            self._writer.add_image(tag, image_tensor, global_step=step)

    def log_hparams(self, hparam_dict: dict, metric_dict: dict):
        if self._writer:
            self._writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        if self._writer:
            self._writer.close()

    # Allow use as a context manager: `with TensorBoardLogger(...) as tb:`
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()