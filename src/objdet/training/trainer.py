"""
training/trainer.py

Main training and validation loops.

Design:
  - Trainer is a simple class (not a framework) — easy to read and debug.
  - Faster R-CNN returns a loss dict during training; we sum the losses.
  - Validation uses torchmetrics / pycocotools via the metrics module.
  - Profiler integration is optional via context-manager injection.
"""

import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ods.entity.config_entity import TrainingPipelineConfig
from ods.evaluation.metrics import COCOEvaluator
from ods.tracking.tensorboard_logger import TensorBoardLogger
from ods.tracking.mlflow_logger import MLflowLogger
from ods.utils.checkpoint import save_checkpoint, load_checkpoint, cleanup_old_checkpoints


class Trainer:
    """
    Encapsulates the training loop for Faster R-CNN.

    Args:
        model:       The Faster R-CNN model.
        train_loader, val_loader: DataLoaders (use custom collate_fn).
        cfg:         Full pipeline config.
        tb_logger:   TensorBoard logger (optional).
        mlf_logger:  MLflow logger (optional).
        profiler:    torch.profiler.profile context (optional, set externally).
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

        # Only optimise parameters that require gradients
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = SGD(
            params,
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=cfg.training.lr_step_size,
            gamma=cfg.training.lr_gamma,
        )

        self.start_epoch = 0
        self.global_step = 0

        # Create checkpoint directory
        Path(cfg.checkpointing.save_dir).mkdir(parents=True, exist_ok=True)

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

            # --- Logging ---
            lr = self.scheduler.get_last_lr()[0]
            print(
                f"[Epoch {epoch+1}] train_loss={train_loss:.4f}  "
                f"mAP={val_metrics.get('map', 0.0):.4f}  lr={lr:.6f}"
            )

            if self.tb_logger:
                self.tb_logger.log_scalar("epoch/train_loss", train_loss, epoch)
                self.tb_logger.log_scalar("epoch/lr", lr, epoch)
                for k, v in val_metrics.items():
                    self.tb_logger.log_scalar(f"epoch/{k}", v, epoch)

            if self.mlf_logger:
                self.mlf_logger.log_metrics(
                    {"train_loss": train_loss, "lr": lr, **val_metrics}, step=epoch
                )

            # --- Checkpointing ---
            if (epoch + 1) % self.cfg.checkpointing.save_every == 0:
                ckpt_path = save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    save_dir=Path(self.cfg.checkpointing.save_dir),
                )
                cleanup_old_checkpoints(
                    Path(self.cfg.checkpointing.save_dir),
                    keep=self.cfg.checkpointing.keep_last,
                )

        print("\n[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one epoch of training; return mean total loss."""
        self.model.train()
        total_loss = 0.0
        log_every = self.cfg.logging.log_every
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Faster R-CNN returns a dict of losses in train mode
            loss_dict = self.model(images, targets)

            # Sum individual losses: classifier, box_reg, objectness, rpn_box_reg
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            # Optional gradient clipping
            if self.cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )

            self.optimizer.step()

            loss_val = losses.item()
            total_loss += loss_val
            self.global_step += 1

            # Periodic batch-level logging
            if (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}]  "
                    f"loss={loss_val:.4f}  ({elapsed:.1f}s)"
                )
                if self.tb_logger:
                    self.tb_logger.log_scalar("train/loss", loss_val, self.global_step)
                    for k, v in loss_dict.items():
                        self.tb_logger.log_scalar(f"train/{k}", v.item(), self.global_step)

        return total_loss / max(len(self.train_loader), 1)

    def _validate(self, epoch: int) -> dict:
        """Run evaluation on the validation set; return metric dict."""
        evaluator = COCOEvaluator(self.device)
        evaluator.evaluate(self.model, self.val_loader)
        metrics = evaluator.get_metrics()
        return metrics