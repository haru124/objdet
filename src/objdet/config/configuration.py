from pathlib import Path
from typing import Optional
import yaml

from objdet.entity.config_entity import (
    DataConfig, ModelConfig, TrainingConfig, LossConfig, EvalConfig,
    CheckpointingConfig, LoggingConfig, ProfilerConfig,
    TrainingPipelineConfig,
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*. Override wins on conflict."""
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
            loss=self._loss_config(r.get("loss", {})),
            eval=self._eval_config(r.get("eval", {})),
            checkpointing=self._checkpointing_config(r.get("checkpointing", {})),
            logging=self._logging_config(r.get("logging", {})),
            profiler=self._profiler_config(r.get("profiler", {})),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _data_config(d: dict) -> DataConfig:
        return DataConfig(
            root=d.get("root", "data/"),
            images_dir=d.get("images_dir", "data/images"),
            annotations_dir=d.get("annotations_dir", "data/gtFine"),
            num_workers=d.get("num_workers", 4),
            pin_memory=d.get("pin_memory", True),
            max_samples=d.get("max_samples", None),
        )

    @staticmethod
    def _model_config(d: dict) -> ModelConfig:
        return ModelConfig(
            num_classes=d.get("num_classes", 9),
            backbone_weights=d.get("backbone_weights", "imagenet"),
            local_weights_path=d.get("local_weights_path", None),
            load_backbone_only=d.get("load_backbone_only", False),
            trainable_backbone_layers=d.get("trainable_backbone_layers", 3),
            min_size=d.get("min_size", 800),
            max_size=d.get("max_size", 1333),
        )

    @staticmethod
    def _training_config(d: dict) -> TrainingConfig:
        return TrainingConfig(
            epochs=d.get("epochs", 20),
            batch_size=d.get("batch_size", 2),
            optimizer=d.get("optimizer", "sgd"),
            learning_rate=d.get("learning_rate", 0.005),
            momentum=d.get("momentum", 0.9),
            weight_decay=d.get("weight_decay", 0.0005),
            lr_scheduler=d.get("lr_scheduler", "step"),
            lr_step_size=d.get("lr_step_size", 7),
            lr_gamma=d.get("lr_gamma", 0.1),
            warmup = d.get("warmup", 0),
            grad_clip=d.get("grad_clip", None),
            device=d.get("device", "cuda"),
            amp=d.get("amp", False),
            accumulation_steps=d.get("accumulation_steps", 1),
            early_stopping = d.get("early_stopping",True),
            early_stopping_patience = d.get("early_stopping_patience", 5),
            early_stopping_min_delta = d.get("early_stopping_min_delta", 0.00001),
            early_stopping_metric = d.get("early_stopping_metric", "map_50_95"),

        )

    @staticmethod
    def _loss_config(d: dict) -> LossConfig:
        return LossConfig(
            classification=d.get("classification", "cross_entropy"),
            box_regression=d.get("box_regression", "smooth_l1"),
            focal_alpha=d.get("focal_alpha", 0.25),
            focal_gamma=d.get("focal_gamma", 2.0),
            smooth_l1_beta=d.get("smooth_l1_beta", 1.0),
            cls_weights=d.get("cls_weights", None),
        )

    @staticmethod
    def _eval_config(d: dict) -> EvalConfig:
        return EvalConfig(
            iou_thresholds=d.get(
                "iou_thresholds", [0.5 + 0.05 * i for i in range(10)]
            ),
            score_threshold=d.get("score_threshold", 0.05),
            max_detections=d.get("max_detections", 100),
        )

    @staticmethod
    def _checkpointing_config(d: dict) -> CheckpointingConfig:
        return CheckpointingConfig(
            save_dir=d.get("save_dir", "outputs/checkpoints/"),
            save_every=d.get("save_every", 2),
            validate_every=d.get("validate_every", 1), 
            keep_last=d.get("keep_last", 3),
        )

    @staticmethod
    def _logging_config(d: dict) -> LoggingConfig:
        return LoggingConfig(
            tensorboard_dir=d.get("tensorboard_dir", "outputs/tensorboard/"),
            mlflow_tracking_uri=d.get("mlflow_tracking_uri", "outputs/mlruns/"),
            log_every=d.get("log_every", 50),
        )

    @staticmethod
    def _profiler_config(d: dict) -> ProfilerConfig:
        return ProfilerConfig(
            enabled=d.get("enabled", False),
            wait=d.get("wait", 1),
            warmup=d.get("warmup", 1),
            active=d.get("active", 3),
            output_dir=d.get("output_dir", "outputs/profiler/"),
        )

