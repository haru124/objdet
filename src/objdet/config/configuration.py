#@staticmethod = a function inside a class that does NOT use class or object state, but is grouped there for organization.
"""
config/configuration.py

Loads config.yaml (base) then deep-merges an optional experiment YAML on top.
Returns a fully populated TrainingPipelineConfig dataclass.

Usage:
    cfg = ConfigurationManager("config/config.yaml",
                               "config/experiments/exp_01.yaml").get_config()
"""

from pathlib import Path
from typing import Optional
import yaml

from ods.entity.config_entity import (
    DataConfig, ModelConfig, TrainingConfig,
    CheckpointingConfig, LoggingConfig, ProfilerConfig,
    TrainingPipelineConfig,
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class ConfigurationManager:
    """
    Reads YAML configs and maps them to typed dataclasses.

    Args:
        base_config_path:    Path to config/config.yaml
        experiment_config_path: Optional path to an experiment override YAML
    """

    def __init__(
        self,
        base_config_path: str | Path = "config/config.yaml",
        experiment_config_path: Optional[str | Path] = None,
    ):
        raw = _load_yaml(Path(base_config_path))

        if experiment_config_path is not None:
            exp_raw = _load_yaml(Path(experiment_config_path))
            raw = _deep_merge(raw, exp_raw)

        self._raw = raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> TrainingPipelineConfig:
        r = self._raw
        return TrainingPipelineConfig(
            project_name=r.get("project_name", "faster_rcnn_cityscapes"),
            experiment_name=r.get("experiment_name", "baseline"),
            data=self._data_config(r.get("data", {})),
            model=self._model_config(r.get("model", {})),
            training=self._training_config(r.get("training", {})),
            checkpointing=self._checkpointing_config(r.get("checkpointing", {})),
            logging=self._logging_config(r.get("logging", {})),
            profiler=self._profiler_config(r.get("profiler", {})),
        )

    # ------------------------------------------------------------------
    # Private helpers — one per sub-config section
    # ------------------------------------------------------------------

    @staticmethod
    def _data_config(d: dict) -> DataConfig:
        cfg = DataConfig(
            root=d.get("root", "data/"),
            images_dir=d.get("images_dir", "data/Images"),
            annotations_dir=d.get("annotations_dir", "data/gtFine"),
            num_workers=d.get("num_workers", 4),
            pin_memory=d.get("pin_memory", True),
        )
        return cfg

    @staticmethod
    def _model_config(d: dict) -> ModelConfig:
        return ModelConfig(
            num_classes=d.get("num_classes", 9),
            pretrained_backbone=d.get("pretrained_backbone", True),
            trainable_backbone_layers=d.get("trainable_backbone_layers", 3),
            min_size=d.get("min_size", 800),
            max_size=d.get("max_size", 1333),
        )

    @staticmethod
    def _training_config(d: dict) -> TrainingConfig:
        return TrainingConfig(
            epochs=d.get("epochs", 20),
            batch_size=d.get("batch_size", 2),
            learning_rate=d.get("learning_rate", 0.005),
            momentum=d.get("momentum", 0.9),
            weight_decay=d.get("weight_decay", 0.0005),
            lr_step_size=d.get("lr_step_size", 7),
            lr_gamma=d.get("lr_gamma", 0.1),
            grad_clip=d.get("grad_clip", None),
            device=d.get("device", "cuda"),
        )

    @staticmethod
    def _checkpointing_config(d: dict) -> CheckpointingConfig:
        return CheckpointingConfig(
            save_dir=d.get("save_dir", "checkpoints/"),
            save_every=d.get("save_every", 2),
            keep_last=d.get("keep_last", 3),
        )

    @staticmethod
    def _logging_config(d: dict) -> LoggingConfig:
        return LoggingConfig(
            tensorboard_dir=d.get("tensorboard_dir", "runs/"),
            mlflow_tracking_uri=d.get("mlflow_tracking_uri", "mlruns/"),
            log_every=d.get("log_every", 50),
        )

    @staticmethod
    def _profiler_config(d: dict) -> ProfilerConfig:
        return ProfilerConfig(
            enabled=d.get("enabled", False),
            wait=d.get("wait", 1),
            warmup=d.get("warmup", 1),
            active=d.get("active", 3),
            output_dir=d.get("output_dir", "profiler_output/"),
        )