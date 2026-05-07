"""
utils/checkpoint.py

Save and load model checkpoints.

Checkpoint files are named  checkpoint_epoch_{N:04d}.pth  so they sort
lexicographically in the correct order, making it easy to find the latest.
"""

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    save_dir: Path,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save model + optimiser + scheduler state to a .pth file.

    Returns the path of the saved file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pth"

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if extra:
        state.update(extra)

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved → {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> int:
    """
    Load model (and optionally optimiser/scheduler) state from a checkpoint.

    Returns:
        epoch: the epoch number stored in the checkpoint (use as start_epoch).
    """
    state = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"])
    print(f"[Checkpoint] Loaded model weights from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state.get("epoch", 0)


def get_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Return the path to the most recent checkpoint, or None if none exist."""
    ckpts = sorted(save_dir.glob("checkpoint_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def cleanup_old_checkpoints(save_dir: Path, keep: int = 3):
    """
    Delete old checkpoint files, keeping only the *keep* most recent.

    Args:
        save_dir: Directory containing checkpoint files.
        keep:     Number of most recent checkpoints to retain.
    """
    ckpts = sorted(save_dir.glob("checkpoint_epoch_*.pth"))
    to_delete = ckpts[: max(0, len(ckpts) - keep)]
    for p in to_delete:
        p.unlink()
        print(f"[Checkpoint] Removed old checkpoint: {p.name}")