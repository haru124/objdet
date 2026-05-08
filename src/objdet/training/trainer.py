"""
training/trainer.py

Main training and validation loops.
Now supports:
  - Configurable optimizer: SGD | Adam | AdamW
  - Configurable LR scheduler: StepLR | CosineAnnealingLR | none
  - Configurable losses via losses.py patching
  - Experiment-isolated checkpoint paths
  - Improved checkpoint naming with epoch + loss
"""

import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from objdet.entity.config_entity import TrainingPipelineConfig
from objdet.evaluation.metrics import COCOEvaluator
from objdet.tracking.tensorboard_logger import TensorBoardLogger
from objdet.tracking.mlflow_logger import MLflowLogger
from objdet.utils.checkpoint import (
    save_checkpoint, load_checkpoint,
    cleanup_old_checkpoints,
)


def build_optimizer(model, training_cfg):
    """
    Factory for optimizer construction.
    Only parameters with requires_grad=True are passed to the optimizer.
    This respects the backbone freeze settings applied by build_backbone().

    SGD  : standard Faster R-CNN default, needs momentum tuning
    Adam : adaptive LR, good for fine-tuning, lower lr needed (~1e-4)
    AdamW: Adam + decoupled weight decay, state-of-the-art for fine-tuning
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = training_cfg.optimizer.lower()

    if name == "sgd":
        optimizer = SGD(
            params,
            lr=training_cfg.learning_rate,
            momentum=training_cfg.momentum,
            weight_decay=training_cfg.weight_decay,
        )
    elif name == "adam":
        optimizer = Adam(
            params,
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
    elif name == "adamw":
        optimizer = AdamW(
            params,
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Choose: 'sgd' | 'adam' | 'adamw'."
        )

    print(
        f"[Optimizer] {name.upper()} | lr={training_cfg.learning_rate} | "
        f"wd={training_cfg.weight_decay} | "
        f"trainable params={sum(p.numel() for p in params):,}"
    )
    return optimizer


def build_scheduler(optimizer, training_cfg):
    """
    Factory for LR scheduler construction.

    StepLR   : decay by gamma every lr_step_size epochs (original Faster R-CNN)
    Cosine   : smooth cosine annealing over all epochs (better for AdamW)
    none     : constant learning rate
    """
    name = training_cfg.lr_scheduler.lower()

    if name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=training_cfg.lr_step_size,
            gamma=training_cfg.lr_gamma,
        )
        print(
            f"[Scheduler] StepLR | step={training_cfg.lr_step_size} | "
            f"gamma={training_cfg.lr_gamma}"
        )

    elif name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.epochs,
            eta_min=training_cfg.learning_rate * 0.01,  # min lr = 1% of initial
        )
        print(f"[Scheduler] CosineAnnealingLR | T_max={training_cfg.epochs}")

    elif name == "none":
        # ConstantLR with factor=1 is effectively no scheduling
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
        print("[Scheduler] No LR scheduling (constant LR).")

    else:
        raise ValueError(
            f"Unknown lr_scheduler: '{name}'. Choose: 'step' | 'cosine' | 'none'."
        )

    return scheduler


class Trainer:
    """
    Encapsulates the training loop for Faster R-CNN.

    Checkpoints are saved under:
        {save_dir}/{experiment_name}/checkpoint_{experiment_name}_epoch_{N:04d}_loss_{L:.4f}.pth

    This isolates outputs per experiment automatically.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingPipelineConfig,
        tb_logger: Optional[TensorBoardLogger] = None,
        mlf_logger: Optional[MLflowLogger] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.tb_logger = tb_logger
        self.mlf_logger = mlf_logger

        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = build_optimizer(model, cfg.training)
        self.scheduler = build_scheduler(self.optimizer, cfg.training)

        self.start_epoch = 0
        self.global_step = 0

        # Experiment-isolated checkpoint directory:
        # outputs/checkpoints/exp_01_smoke_test/
        self.ckpt_dir = (
            Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Trainer] Checkpoints will be saved to: {self.ckpt_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: str | Path):
        """Load state from a checkpoint to resume interrupted training."""
        self.start_epoch = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        print(f"[Trainer] Resumed from epoch {self.start_epoch}")

    def fit(self):
        """Run the full training loop from start_epoch to cfg.training.epochs."""
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            print(f"\n=== Epoch {epoch + 1} / {self.cfg.training.epochs} ===")

            train_loss = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch+1}] train_loss={train_loss:.4f} "
                f"mAP={val_metrics.get('map', 0.0):.4f} "
                f"mAP@50={val_metrics.get('map_50', 0.0):.4f} "
                f"lr={lr:.6f}"
            )

            if self.tb_logger:
                self.tb_logger.log_scalar("epoch/train_loss", train_loss, epoch)
                self.tb_logger.log_scalar("epoch/lr", lr, epoch)
                for k, v in val_metrics.items():
                    self.tb_logger.log_scalar(f"epoch/{k}", v, epoch)

            if self.mlf_logger:
                self.mlf_logger.log_metrics(
                    {"train_loss": train_loss, "lr": lr, **val_metrics},
                    step=epoch,
                )

            # --- Checkpointing ---
            if (epoch + 1) % self.cfg.checkpointing.save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=train_loss,
                    experiment_name=self.cfg.experiment_name,
                    save_dir=self.ckpt_dir,
                )
                cleanup_old_checkpoints(
                    self.ckpt_dir,
                    keep=self.cfg.checkpointing.keep_last,
                )

        print("\n[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one epoch; return mean total loss."""
        self.model.train()
        total_loss = 0.0
        log_every = self.cfg.logging.log_every
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            # Faster R-CNN returns a dict of losses in train mode:
            # {loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg}
            # If losses have been patched via patch_roi_head_losses(),
            # loss_classifier and loss_box_reg use our custom functions.
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            if self.cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )

            self.optimizer.step()

            loss_val = losses.item()
            total_loss += loss_val
            self.global_step += 1

            if (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss_val:.4f} "
                    f"(cls={loss_dict.get('loss_classifier', torch.tensor(0)).item():.4f} "
                    f"box={loss_dict.get('loss_box_reg', torch.tensor(0)).item():.4f} "
                    f"obj={loss_dict.get('loss_objectness', torch.tensor(0)).item():.4f} "
                    f"rpn_box={loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.4f}) "
                    f"[{elapsed:.1f}s]"
                )
                if self.tb_logger:
                    self.tb_logger.log_scalar("train/total_loss", loss_val, self.global_step)
                    for k, v in loss_dict.items():
                        self.tb_logger.log_scalar(f"train/{k}", v.item(), self.global_step)

        return total_loss / max(len(self.train_loader), 1)

    def _validate(self, epoch: int) -> dict:
        """Run evaluation on the validation set; return metric dict."""
        evaluator = COCOEvaluator(self.device, self.cfg.eval)
        evaluator.evaluate(self.model, self.val_loader)
        return evaluator.get_metrics()