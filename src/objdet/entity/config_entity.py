"""
entity/config_entity.py

Typed dataclasses that mirror the YAML config structure.
Using dataclasses instead of raw dicts gives IDE auto-complete and
avoids key-name typos at runtime.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    root: str = "data/"
    images_dir: str = "data/Images"
    annotations_dir: str = "data/gtFine"
    num_workers: int = 4
    pin_memory: bool = True

    # Resolved Path helpers (populated by ConfigurationManager)
    images_path: Path = field(default_factory=Path, init=False) #default_factory is used to specify a factory function that will be called to generate a default value for the field when no value is provided during instantiation. In this case, it creates a new Path object using the default images_dir value.
    annotations_path: Path = field(default_factory=Path, init=False)

    def __post_init__(self):
        self.images_path = Path(self.images_dir)
        self.annotations_path = Path(self.annotations_dir)


@dataclass
class ModelConfig:
    num_classes: int = 9
    pretrained_backbone: bool = True
    trainable_backbone_layers: int = 3
    min_size: int = 800
    max_size: int = 1333


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 7
    lr_gamma: float = 0.1
    grad_clip: Optional[float] = None
    device: str = "cuda"


@dataclass
class CheckpointingConfig:
    save_dir: str = "checkpoints/"
    save_every: int = 2
    keep_last: int = 3

    save_path: Path = field(default_factory=Path, init=False)

    def __post_init__(self):
        self.save_path = Path(self.save_dir)


@dataclass
class LoggingConfig:
    tensorboard_dir: str = "runs/"
    mlflow_tracking_uri: str = "mlruns/"
    log_every: int = 50


@dataclass
class ProfilerConfig:
    enabled: bool = False
    wait: int = 1
    warmup: int = 1
    active: int = 3
    output_dir: str = "profiler_output/"


@dataclass
class TrainingPipelineConfig:
    """Aggregates all sub-configs into one object passed around the pipeline."""
    project_name: str = "faster_rcnn_cityscapes"
    experiment_name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig) #default_factory is used to specify a factory function that will be called to generate a default value for the field when no value is provided during instantiation. In this case, it creates a new DataConfig object using the default values defined in the DataConfig dataclass.
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)