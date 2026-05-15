"""
tracking/tensorboard_logger.py

Folder structure:
  outputs/tensorboard/
    exp_01_sgd_cross_entropy_smoothl1/
      train_run_20250515_143022/     ← training run 1
        events.out.tfevents.*
      train_run_20250515_160045/     ← training run 2 (re-run)
        events.out.tfevents.*
      inference_20250515_171200/     ← inference logged here
        events.out.tfevents.*

        
Each execution creates a unique subdirectory so runs are never mixed.
TensorBoard shows each as a separate selectable entry in the left panel.

To view results, run `tensorboard --logdir outputs/tensorboard` and open the
python -m tensorboard.main --logdir outputs/tensorboard --port 6006

"""

from datetime import datetime
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TensorBoardLogger:
    """
    Args:
        log_dir:         Root directory e.g. "outputs/tensorboard"
        experiment_name: Config name e.g. "exp_01_sgd_cross_entropy_smoothl1"
        run_type:        "train" or "inference" — used as folder name prefix
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str = "run",
        run_type: str = "train",
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_dir) / experiment_name / f"{run_type}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

        if _TB_AVAILABLE:
            self._writer = SummaryWriter(log_dir=str(run_dir))
            print(f"[TensorBoard] Logging to: {run_dir}")
        else:
            self._writer = None
            print("[TensorBoard] tensorboard not installed — logging disabled.")

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def log_scalar(self, tag: str, value: float, step: int):
        if self._writer:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag: str, value_dict: dict[str, float], step: int):
        if self._writer:
            self._writer.add_scalars(tag, value_dict, global_step=step)

    def log_image(self, tag: str, image_tensor, step: int):
        if self._writer:
            self._writer.add_image(tag, image_tensor, global_step=step)

    def log_hparams(self, hparam_dict: dict, metric_dict: dict):
        if self._writer:
            self._writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        if self._writer:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()