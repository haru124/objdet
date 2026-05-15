"""
tracking/mlflow_logger.py

Single-run design:
  - main.py creates ONE MLflowLogger and passes it to both Trainer and inference.
  - Training metrics (train_*, val_*) and test metrics (test_*) all log to the
    same run, so one MLflow run = one full experiment execution.

Run lifecycle rules:
  - If a RUNNING run exists with this name (crashed previously): RESUME it.
    Metrics already logged are preserved. Training continues from checkpoint.
  - If a FINISHED run exists: start a NEW run (new execution for comparison).
  - If no run exists: start a new run.


Caller (main.py) is responsible for calling end_run() via try/finally.

  
To view results, run `mlflow ui` pointing to the tracking URI. Each run is listed
python -m mlflow ui `
    --backend-store-uri "file:///C:/Users/BTI-002006/OneDrive - BMW Techworks India Private Limited/Projects/objdet/outputs/mlruns" `
    --port 5000  

"""

from pathlib import Path
from typing import Any, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class MLflowLogger:
    def __init__(
        self,
        tracking_uri: str | Path = "outputs/mlruns",
        experiment_name: str = "faster_rcnn",
        run_name: Optional[str] = None,
    ):
        self._active = False
        self._run = None

        if not _MLFLOW_AVAILABLE:
            print("[MLflow] mlflow not installed — tracking disabled.")
            return

        mlflow.set_tracking_uri(str(tracking_uri))
        experiment = mlflow.set_experiment(experiment_name)
        client = MlflowClient()

        # ── Search for existing run with this name ────────────────────
        resume_run_id = None
        if run_name:
            existing = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if existing:
                latest = existing[0]
                if latest.info.status == "RUNNING":
                    # Crashed run — resume it so metrics are continuous
                    resume_run_id = latest.info.run_id
                    print(f"[MLflow] Resuming crashed run: {resume_run_id[:8]}...")
                elif latest.info.status == "FINISHED":
                    # Previous run completed cleanly — new execution = new run
                    print(f"[MLflow] Previous run FINISHED. Starting fresh run.")

        self._run = mlflow.start_run(
            run_id=resume_run_id,  # None = create new, str = resume existing
            run_name=run_name,     # only used when creating new
        )
        self._active = True
        action = "Resumed" if resume_run_id else "Started"
        print(f"[MLflow] {action} | experiment={experiment_name} | "
              f"run={run_name} | id={self._run.info.run_id[:8]}...")

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None

    def log_params(self, params: dict[str, Any]):
        """
        Log hyperparameters. Safe to call on resumed runs —
        skips params that already exist with the same value.
        """
        if not self._active:
            return
        client = MlflowClient()
        for key, value in params.items():
            try:
                client.log_param(self._run.info.run_id, key, value)
            except mlflow.exceptions.MlflowException:
                pass  # param already logged with same value on resume

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        if not self._active:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"[MLflow] log_metrics warning: {e}")

    def log_artifact(self, local_path: str | Path):
        if not self._active:
            return
        try:
            mlflow.log_artifact(str(local_path))
        except Exception as e:
            print(f"[MLflow] log_artifact warning: {e}")

    def set_tags(self, tags: dict[str, str]):
        if self._active:
            mlflow.set_tags(tags)

    def end_run(self):
        if self._active:
            mlflow.end_run()
            self._active = False
            print("[MLflow] Run ended → FINISHED")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()