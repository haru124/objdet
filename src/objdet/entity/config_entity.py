from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    root: str = "data/"
    images_dir: str = "data/images"
    annotations_dir: str = "data/gtFine"
    num_workers: int = 4
    pin_memory: bool = True
    max_samples: Optional[int] = None   # For debugging: limit number of samples (images) to load, set to null for no limit

    # Resolved paths — populated by __post_init__, NOT from YAML directly
    images_path: Path = field(default_factory=Path, init=False)
    annotations_path: Path = field(default_factory=Path, init=False)

    def __post_init__(self):
        self.images_path = Path(self.images_dir)
        self.annotations_path = Path(self.annotations_dir)


# ---------------------------------------------------------------------------
# MODEL
# backbone_weights: "imagenet" | "coco" | "local" | "none"
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    num_classes: int = 9
    backbone_weights: str = "local"      # replaces old pretrained_backbone bool
    local_weights_path: Optional[str] = None
    load_backbone_only: bool = False        # True → only load backbone from local file
    trainable_backbone_layers: int = 3
    min_size: int = 800
    max_size: int = 1333

#field() gives extra control over a dataclass attribute -- to create customized field
#field() lets you specify:
#  default values -- dataclasses require a value for fields unless: they are initialized later or excluded from init
#####  provides a placeholder/default object.
#  factories - 
####  x: list = []
####  Problem: same list shared across all instances
####  Correct way:
####  x: list = field(default_factory=list)

#  whether included in constructor
#  repr behavior
#  comparison behavior
#  metadata



# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 2
    optimizer: str = "sgd"              # "sgd" | "adam" | "adamw"
    learning_rate: float = 0.005
    momentum: float = 0.9               # SGD only
    weight_decay: float = 0.0005
    lr_scheduler: str = "step"          # "step" | "cosine" | "none"
    lr_step_size: int = 7
    lr_gamma: float = 0.1
    grad_clip: Optional[float] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    classification: str = "cross_entropy"   # "cross_entropy" | "focal"
    box_regression: str = "smooth_l1"       # "smooth_l1" | "l1" | "giou" | "diou" | "ciou"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    smooth_l1_beta: float = 1.0


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5 + 0.05 * i for i in range(10)]
    )
    score_threshold: float = 0.05
    max_detections: int = 100


# ---------------------------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------------------------
@dataclass
class CheckpointingConfig:
    save_dir: str = "outputs/checkpoints/"
    save_every: int = 2
    keep_last: int = 3
    save_path: Path = field(default_factory=Path, init=False)

    def __post_init__(self):
        self.save_path = Path(self.save_dir)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    tensorboard_dir: str = "outputs/tensorboard/"
    mlflow_tracking_uri: str = "outputs/mlruns/"
    log_every: int = 50


# ---------------------------------------------------------------------------
# PROFILER
# ---------------------------------------------------------------------------
@dataclass
class ProfilerConfig:
    enabled: bool = False
    wait: int = 1
    warmup: int = 1
    active: int = 3
    output_dir: str = "outputs/profiler/"


# ---------------------------------------------------------------------------
# TOP-LEVEL PIPELINE CONFIG
# ---------------------------------------------------------------------------
@dataclass
class TrainingPipelineConfig:
    project_name: str = "faster_rcnn_cityscapes"
    experiment_name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    