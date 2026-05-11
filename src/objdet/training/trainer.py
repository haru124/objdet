"""
training/trainer.py

supports:
  - Configurable optimizer: SGD | Adam | AdamW
  - Configurable LR scheduler: StepLR | CosineAnnealingLR | none
  - Configurable losses via losses.py patching
  - Experiment-isolated checkpoint paths
  - Improved checkpoint naming with epoch + loss

Tracks and saves:
  - Per-batch: total loss + 4 component losses
  - Per-epoch: mean of all 5 loss values (train)
  - Per-eval-epoch: validation loss + validation mAP
  - History dict saved to checkpoint for inference plotting
"""

import time
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from objdet.entity.config_entity import TrainingPipelineConfig
from objdet.evaluation.metrics import COCOEvaluator
from objdet.tracking.tensorboard_logger import TensorBoardLogger
from objdet.tracking.mlflow_logger import MLflowLogger
from objdet.utils.checkpoint import save_checkpoint, load_checkpoint, cleanup_old_checkpoints


def build_optimizer(model, training_cfg):
    """
    Build optimizer from config.
    SGD   → standard Faster R-CNN default
    Adam  → adaptive LR, lower lr ~1e-4
    AdamW → Adam + decoupled weight decay (best for fine-tuning transformers/FPN)
    """
    from torch.optim import SGD, Adam, AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    name = training_cfg.optimizer.lower()

    if name == "sgd":
        opt = SGD(params, lr=training_cfg.learning_rate,
                  momentum=training_cfg.momentum, weight_decay=training_cfg.weight_decay)
    elif name == "adam":
        opt = Adam(params, lr=training_cfg.learning_rate,
                   weight_decay=training_cfg.weight_decay)
    elif name == "adamw":
        opt = AdamW(params, lr=training_cfg.learning_rate,
                    weight_decay=training_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    print(f"[Optimizer] {name.upper()} | lr={training_cfg.learning_rate} | "
          f"wd={training_cfg.weight_decay} | params={sum(p.numel() for p in params):,}")
    return opt


def build_scheduler(optimizer, training_cfg):
    """Build LR scheduler from config."""
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ConstantLR
    name = training_cfg.lr_scheduler.lower()

    if name == "step":
        sched = StepLR(optimizer, step_size=training_cfg.lr_step_size,
                       gamma=training_cfg.lr_gamma)
    elif name == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=training_cfg.epochs,
                                  eta_min=training_cfg.learning_rate * 0.01)
    elif name == "none":
        sched = ConstantLR(optimizer, factor=1.0, total_iters=0)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return sched


class Trainer:
    """
    Training loop with full loss history tracking.

    History structure (self.history):
    {
        "train": {
            "epoch":              [1, 2, 3, ...],
            "total_loss":         [0.85, 0.72, ...],
            "loss_classifier":    [0.30, 0.25, ...],
            "loss_box_reg":       [0.20, 0.18, ...],
            "loss_objectness":    [0.25, 0.20, ...],
            "loss_rpn_box_reg":   [0.10, 0.09, ...],
        },
        "val": {
            "epoch":              [2, 4, 6, ...],    # eval every save_every epochs
            "total_loss":         [0.90, 0.80, ...],
            "loss_classifier":    [...],
            "loss_box_reg":       [...],
            "loss_objectness":    [...],
            "loss_rpn_box_reg":   [...],
            "map":                [0.32, 0.38, ...],
            "map_50":             [0.55, 0.60, ...],
            "map_75":             [0.28, 0.34, ...],
            "ap_per_class":       [{...}, {...}, ...],  # per-epoch class APs
            
        }
    }
    This dict is saved to:
      outputs/checkpoints/{experiment_name}/training_history.json
    so inference.py can load and plot without retraining.
    """

    LOSS_KEYS = [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]

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

        # Experiment-isolated checkpoint dir: outputs/checkpoints/exp_01/
        self.ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # History dict — persisted to JSON after every epoch
        self.history = {
            "train": {k: [] for k in ["epoch", "total_loss"] + self.LOSS_KEYS},
            "val":   {k: [] for k in ["epoch", "total_loss"] + self.LOSS_KEYS +
                      ["map", "map_50", "map_75", "ap_per_class","precision", "recall"]},
        }
        self.history_path = self.ckpt_dir / "training_history.json"

        print(f"[Trainer] Device       : {self.device}")
        print(f"[Trainer] Checkpoints  : {self.ckpt_dir}")
        print(f"[Trainer] History file : {self.history_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: str | Path):
        """Load checkpoint and restore history if available."""
        self.start_epoch = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        # Restore history so plots are continuous across resumed runs
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
            print(f"[Trainer] Restored training history from {self.history_path}")
        print(f"[Trainer] Resumed from epoch {self.start_epoch}")

    def fit(self):
        """Full training loop."""
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} / {self.cfg.training.epochs}")
            print(f"{'='*60}")

            # ── Train ─────────────────────────────────────────────────
            train_losses = self._train_one_epoch(epoch)
            # train_losses = {"total_loss": float, "loss_classifier": float, ...}

            self.history["train"]["epoch"].append(epoch + 1)
            self.history["train"]["total_loss"].append(train_losses["total_loss"])
            for k in self.LOSS_KEYS:
                self.history["train"][k].append(train_losses.get(k, 0.0))

            # ── Validate ───────────────────────────────────────────────
            # Run validation every save_every epochs to save compute.
            # Always validate on last epoch.
            run_val = (
                (epoch + 1) % self.cfg.checkpointing.save_every == 0
                or (epoch + 1) == self.cfg.training.epochs
            )

            if run_val:
                val_losses, val_metrics = self._validate(epoch)
                # val_losses = same structure as train_losses
                # val_metrics = {"map": float, "map_50": float, "map_75": float,
                #                "ap_per_class": {class_name: float, ...}}

                self.history["val"]["epoch"].append(epoch + 1)
                self.history["val"]["total_loss"].append(val_losses["total_loss"])
              
                for k in self.LOSS_KEYS:
                    self.history["val"][k].append(val_losses.get(k, 0.0))
                self.history["val"]["map"].append(val_metrics.get("map", 0.0))
                self.history["val"]["map_50"].append(val_metrics.get("map_50", 0.0))
                self.history["val"]["map_75"].append(val_metrics.get("map_75", 0.0))
                self.history["val"]["ap_per_class"].append(
                    val_metrics.get("ap_per_class", {})
                )
                self.history["val"]["precision"].append(val_metrics.get("precision", 0.0))
                self.history["val"]["recall"].append(val_metrics.get("recall", 0.0))
            else:
                val_losses = {"total_loss": float("nan")}
                val_metrics = {"map": float("nan")}

            # ── LR step ────────────────────────────────────────────────
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # ── Console summary ────────────────────────────────────────
            print(
                f"\n[Epoch {epoch+1}] "
                f"lr = {lr:.6f} | "
                f"total_train_loss={train_losses['total_loss']:.4f} | "
                f"cls={train_losses.get('loss_classifier', 0):.4f} | "
                f"box={train_losses.get('loss_box_reg', 0):.4f} | "
                f"obj={train_losses.get('loss_objectness', 0):.4f} | "
                f"rpn_box={train_losses.get('loss_rpn_box_reg', 0):.4f}"
            )
            if run_val:
                print(
                    f"[Epoch {epoch+1}] "
                    f"total_val_loss={val_losses['total_loss']:.4f} | "
                    f"mAP={val_metrics.get('map', 0):.4f} | "
                    f"mAP@50={val_metrics.get('map_50', 0):.4f} | "
                    f"mAP@75={val_metrics.get('map_75', 0):.4f} | "
                    f"lr={lr:.6f}"
                )
                # Per-class AP
                if "ap_per_class" in val_metrics:
                    print("[Epoch {e}] Per-class AP:".format(e=epoch+1))
                    for cls_name, ap in val_metrics["ap_per_class"].items():
                        print(f"  {cls_name:<15}: {ap:.4f}")

            # ── TensorBoard ────────────────────────────────────────────
            if self.tb_logger:
                self.tb_logger.log_scalar(
                    "epoch/train_total_loss", train_losses["total_loss"], epoch
                )
                for k in self.LOSS_KEYS:
                    self.tb_logger.log_scalar(
                        f"epoch/train_{k}", train_losses.get(k, 0.0), epoch
                    )
                self.tb_logger.log_scalar("epoch/lr", lr, epoch)
                if run_val:
                    self.tb_logger.log_scalar(
                        "epoch/val_total_loss", val_losses["total_loss"], epoch
                    )
                    for k in self.LOSS_KEYS:
                        self.tb_logger.log_scalar(
                            f"epoch/val_{k}", val_losses.get(k, 0.0), epoch
                        )
                    for metric_name, v in val_metrics.items():
                        if isinstance(v, float):
                            self.tb_logger.log_scalar(
                                f"epoch/val_{metric_name}", v, epoch
                            )

            # ── MLflow ─────────────────────────────────────────────────
            if self.mlf_logger:
                metrics_to_log = {
                    "train_total_loss": train_losses["total_loss"], "lr": lr
                }
                metrics_to_log.update(
                    {f"train_{k}": train_losses.get(k, 0.0) for k in self.LOSS_KEYS}
                )
                if run_val:
                    metrics_to_log["val_total_loss"] = val_losses["total_loss"]
                    metrics_to_log.update(
                        {f"val_{k}": val_losses.get(k, 0.0) for k in self.LOSS_KEYS}
                    )
                    metrics_to_log.update(
                        {k: v for k, v in val_metrics.items() if isinstance(v, float)}
                    )
                self.mlf_logger.log_metrics(metrics_to_log, step=epoch)

            # ── Persist history ────────────────────────────────────────
            # Save after every epoch so crashes don't lose history
            self._save_history()

            # ── Checkpoint ────────────────────────────────────────────
            if (epoch + 1) % self.cfg.checkpointing.save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=train_losses["total_loss"],
                    experiment_name=self.cfg.experiment_name,
                    save_dir=self.ckpt_dir,
                    extra={"history": self.history},   # embed history in checkpoint too
                )
                cleanup_old_checkpoints(
                    self.ckpt_dir, keep=self.cfg.checkpointing.keep_last
                )

        print("\n[Trainer] Training complete.")
        print(f"[Trainer] History saved to: {self.history_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict:
        """
        One epoch of training.

        Returns dict of mean losses over all batches:
        {
            "total_loss":       float,
            "loss_classifier":  float,
            "loss_box_reg":     float,
            "loss_objectness":  float,
            "loss_rpn_box_reg": float,
        }
        """
        self.model.train()

        # Accumulators for each loss component
        accum = {k: 0.0 for k in ["total_loss"] + self.LOSS_KEYS}
        log_every = self.cfg.logging.log_every
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]

            # Faster R-CNN train mode returns:
            # {
            #   "loss_classifier":  Tensor  ← ROI classification loss
            #   "loss_box_reg":     Tensor  ← ROI box regression loss
            #   "loss_objectness":  Tensor  ← RPN objectness loss
            #   "loss_rpn_box_reg": Tensor  ← RPN box regression loss
            # }
            loss_dict = self.model(images, targets)
            total_loss = sum(v for v in loss_dict.values())

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )
            self.optimizer.step()

            # Accumulate all losses
            accum["total_loss"] += total_loss.item()
            for k in self.LOSS_KEYS:
                accum[k] += loss_dict.get(k, torch.tensor(0.0)).item()

            self.global_step += 1

            if (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{batch_idx+1:>4}/{len(self.train_loader)}] "
                    f"total={total_loss.item():.4f} | "
                    f"cls={loss_dict.get('loss_classifier', torch.tensor(0)).item():.4f} | "
                    f"box={loss_dict.get('loss_box_reg', torch.tensor(0)).item():.4f} | "
                    f"obj={loss_dict.get('loss_objectness', torch.tensor(0)).item():.4f} | "
                    f"rpn_box={loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.4f} "
                    f"[{elapsed:.1f}s]"
                )
                if self.tb_logger:
                    self.tb_logger.log_scalar(
                        "batch/total_loss", total_loss.item(), self.global_step
                    )
                    for k in self.LOSS_KEYS:
                        self.tb_logger.log_scalar(
                            f"batch/{k}",
                            loss_dict.get(k, torch.tensor(0)).item(),
                            self.global_step,
                        )

        n = max(len(self.train_loader), 1)
        return {k: v / n for k, v in accum.items()}

    def _validate(self, epoch: int) -> tuple[dict, dict]:
        """
        Compute validation loss AND validation mAP.

        Returns:
            val_losses : same structure as _train_one_epoch return
            val_metrics: {"map": float, "map_50": float, "map_75": float,
                          "ap_per_class": {class_name: float}}

        How validation loss is computed:
          We temporarily switch model to TRAIN mode (so it computes loss dict),
          run forward with no_grad, then switch back to eval for mAP computation.
          This gives a true "how well does the model fit the val data" signal.
        """
        # ── Validation loss (model in train mode to get loss dict) ─────
        self.model.train()
        val_accum = {k: 0.0 for k in ["total_loss"] + self.LOSS_KEYS}

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
                loss_dict = self.model(images, targets)
                total_loss = sum(v for v in loss_dict.values())
                val_accum["total_loss"] += total_loss.item()
                for k in self.LOSS_KEYS:
                    val_accum[k] += loss_dict.get(k, torch.tensor(0.0)).item()

        n = max(len(self.val_loader), 1)
        val_losses = {k: v / n for k, v in val_accum.items()}

        # ── Validation mAP (model in eval mode) ───────────────────────
        evaluator = COCOEvaluator(self.device, self.cfg.eval)
        evaluator.evaluate(self.model, self.val_loader)
        val_metrics = evaluator.get_metrics()

        return val_losses, val_metrics

    def _save_history(self):
        """Persist history dict to JSON."""
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)