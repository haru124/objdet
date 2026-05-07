"""
tracking/mlflow_logger.py

Thin wrapper around mlflow to log parameters, metrics, and artifacts.

Design: fail-soft — if mlflow is not installed or the tracking server is
unavailable, methods are no-ops rather than crashing the training loop.
"""

from pathlib import Path
from typing import Any, Optional

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class MLflowLogger:
    """
    Wraps MLflow experiment tracking.

    Args:
        tracking_uri:    Local path or remote URI for the MLflow server.
        experiment_name: Name of the MLflow experiment (created if absent).
        run_name:        Optional descriptive name for this run.
    """

    def __init__(
        self,
        tracking_uri: str | Path = "mlruns/",
        experiment_name: str = "faster_rcnn",
        run_name: Optional[str] = None,
    ):
        self._active = False

        if not _MLFLOW_AVAILABLE:
            print("[MLflow] mlflow not installed — tracking disabled.")
            return

        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._active = True
        print(f"[MLflow] Run started: {self._run.info.run_id}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters (call once before training starts)."""
        if self._active:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        """Log a dict of scalar metrics."""
        if self._active:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path):
        """Log a local file (e.g. checkpoint, config) as an artifact."""
        if self._active:
            mlflow.log_artifact(str(local_path))

    def set_tags(self, tags: dict[str, str]):
        if self._active:
            mlflow.set_tags(tags)

    def end_run(self):
        if self._active:
            mlflow.end_run()
            self._active = False

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()