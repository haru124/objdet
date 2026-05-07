"""
constants/__init__.py

Central place for project-wide constants.
Keeping them here avoids magic strings scattered across modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem roots (relative to the project root, i.e. where main.py lives)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # …/objdet/
CONFIG_DIR   = PROJECT_ROOT / "config"
DATA_ROOT    = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Cityscapes label map
# Only the 8 classes commonly used for detection; index 0 is always background.
# ---------------------------------------------------------------------------
CITYSCAPES_CLASSES = [
    "__background__",   # 0 — required by Faster R-CNN
    "person",           # 1
    "rider",            # 2
    "car",              # 3
    "truck",            # 4
    "bus",              # 5
    "motorcycle",       # 6
    "bicycle",          # 7
    "train",            # 8
]

# Map label name → integer index
LABEL_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CITYSCAPES_CLASSES)}

# Number of classes including background
NUM_CLASSES: int = len(CITYSCAPES_CLASSES)   # 9

# ---------------------------------------------------------------------------
# Dataset split names
# ---------------------------------------------------------------------------
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Cityscapes annotation file suffix
# e.g.  aachen_000000_000019_gtFine_polygons.json
# ---------------------------------------------------------------------------
GT_FINE_SUFFIX = "_gtFine_polygons.json"