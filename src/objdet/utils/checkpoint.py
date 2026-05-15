"""
utils/checkpoint.py

Metric naming convention (internal Python keys):
  map_50_95  ←  mAP@[0.5:0.95]   primary COCO metric, used for best-ckpt tracking
  map_50     ←  mAP@0.50
  map_75     ←  mAP@0.75

Checkpoint filename convention:
  Regular : {exp}_epoch_{N:04d}_loss_{loss:.4f}_map50-95_{map:.4f}.pth
  Best    : best_{exp}_loss_{loss:.4f}_map50-95_{map:.4f}.pth

Best checkpoint selection:
  Primary   : highest val mAP@[0.5:0.95]
  Tie-break : lowest val loss (when mAP is equal to 4 decimal places)

The best checkpoint is NEVER deleted by cleanup_old_checkpoints.
It is only replaced when a strictly better checkpoint is found.
"""

from pathlib import Path
from typing import Optional
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    map_50_95: float,
    experiment_name: str,
    save_dir: Path,
    extra: Optional[dict] = None,
) -> Path:
    """Save a regular epoch checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"{experiment_name}"
        f"_epoch_{epoch:04d}"
        f"_loss_{loss:.4f}"
        f"_map50-95_{map_50_95:.4f}.pth"
    )
    ckpt_path = save_dir / fname

    state = {
        "epoch":                epoch,
        "experiment_name":      experiment_name,
        "loss":                 loss,
        "map_50_95":            map_50_95,       # standardized key
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "backbone_state_dict":  model.backbone.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved → {ckpt_path.name}")
    return ckpt_path


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    map_50_95: float,
    experiment_name: str,
    save_dir: Path,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save (overwrite) the best checkpoint.

    Selection criteria:
      Primary  : highest map_50_95
      Tie-break: lowest loss (when map_50_95 equal to 4dp)

    Always exactly ONE best_*.pth per experiment.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove previous best for this experiment
    for old in save_dir.glob(f"best_{experiment_name}_*.pth"):
        old.unlink()
        print(f"[Checkpoint] Replaced old best: {old.name}")

    fname = (
        f"best_{experiment_name}"
        f"_loss_{loss:.4f}"
        f"_map50-95_{map_50_95:.4f}.pth"
    )
    best_path = save_dir / fname

    state = {
        "epoch":                epoch,
        "experiment_name":      experiment_name,
        "loss":                 loss,
        "map_50_95":            map_50_95,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "backbone_state_dict":  model.backbone.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, best_path)
    print(
        f"[Checkpoint] ★ Best → {best_path.name}  "
        f"(mAP@[.5:.95]={map_50_95:.4f}  loss={loss:.4f})"
    )
    return best_path


def save_backbone_only_checkpoint(model, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state_dict": model.backbone.state_dict()}, save_path)
    print(f"[Checkpoint] Backbone-only saved → {save_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> int:
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    # Support both old key ("map50") and new standardized key ("map_50_95")
    map_val = state.get("map_50_95", state.get("map50", state.get("map", 0.0)))

    print(
        f"[Checkpoint] Loaded: {Path(checkpoint_path).name}\n"
        f"             epoch={state.get('epoch','?')}  "
        f"loss={state.get('loss', 0):.4f}  "
        f"mAP@[.5:.95]={map_val:.4f}"
    )
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state.get("epoch", 0)


def get_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Most recent regular epoch checkpoint (not best_*)."""
    ckpts = sorted(Path(save_dir).glob("*_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def get_best_checkpoint(save_dir: Path) -> Optional[Path]:
    """Best checkpoint if it exists."""
    bests = list(Path(save_dir).glob("best_*.pth"))
    return bests[0] if bests else None


def cleanup_old_checkpoints(save_dir: Path, keep: int = 3):
    """Delete old regular checkpoints. Never touches best_*.pth."""
    ckpts = sorted(Path(save_dir).glob("*_epoch_*.pth"))
    for p in ckpts[: max(0, len(ckpts) - keep)]:
        p.unlink()
        print(f"[Checkpoint] Removed: {p.name}")