"""
utils/checkpoint.py

Checkpoint naming: {experiment_name}_epoch_{N:04d}_loss_{loss:.4f}.pth
Supports:
  - full model checkpoint (model + optimizer + scheduler)
  - backbone-only checkpoint (for transfer learning from another task)
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
    experiment_name: str,
    save_dir: Path,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a full training checkpoint.

    File name: {experiment_name}_epoch_{epoch:04d}_loss_{loss:.4f}.pth
    Saved under: save_dir/ (experiment sub-folder created by Trainer)

    State dict keys:
        epoch               → int
        experiment_name     → str
        loss                → float
        model_state_dict    → model weights
        optimizer_state_dict→ optimizer state (momentum buffers etc.)
        scheduler_state_dict→ scheduler state
        backbone_state_dict → backbone sub-module weights (convenience key
                              for loading backbone-only in a future run)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{experiment_name}_epoch_{epoch:04d}_loss_{loss:.4f}.pth"
    ckpt_path = save_dir / fname

    state = {
        "epoch": epoch,
        "experiment_name": experiment_name,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        # Save backbone separately so future experiments can load backbone-only
        "backbone_state_dict": model.backbone.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved → {ckpt_path}")
    return ckpt_path


def save_backbone_only_checkpoint(
    model: torch.nn.Module,
    save_path: str | Path,
):
    """
    Save ONLY the backbone weights to a separate file.
    Useful for:
      - Sharing a pretrained backbone between experiments
      - Loading into a fresh detector via backbone_weights: "local"
        with load_backbone_only: true

    File format: {"backbone_state_dict": state_dict}
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state_dict": model.backbone.state_dict()}, save_path)
    print(f"[Checkpoint] Backbone-only checkpoint saved → {save_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> int:
    """
    Load a full model checkpoint (saved by save_checkpoint above).

    Returns:
        epoch: the epoch stored in the checkpoint (use as start_epoch)
    """
    state = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"])
    print(
        f"[Checkpoint] Loaded model from {checkpoint_path} "
        f"(epoch {state.get('epoch', '?')}, "
        f"loss {state.get('loss', '?'):.4f})"
    )

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state.get("epoch", 0)


def get_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """
    Return the most recent checkpoint in save_dir.
    Checkpoints are named with epoch number so lexicographic sort works.
    """
    ckpts = sorted(save_dir.glob("*_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def cleanup_old_checkpoints(save_dir: Path, keep: int = 3):
    """Delete old checkpoints, keeping only the *keep* most recent."""
    ckpts = sorted(save_dir.glob("*_epoch_*.pth"))
    to_delete = ckpts[: max(0, len(ckpts) - keep)]
    for p in to_delete:
        p.unlink()
        print(f"[Checkpoint] Removed old checkpoint: {p.name}")