"""
training/trainer.py

Metric naming (internal keys, consistent everywhere):
  map_50_95  →  mAP@[0.5:0.95]   primary COCO metric
  map_50     →  mAP@0.50
  map_75     →  mAP@0.75

Best checkpoint selection:
  Primary criterion  : highest val map_50_95
  Tie-break criterion: lowest val loss (when map_50_95 equal to 4dp)

Validation frequency (validate_every) and checkpoint frequency (save_every)
are decoupled. You can validate every epoch but checkpoint every 5.

Note on model.train() during validation loss computation:
  torchvision Faster R-CNN only returns a loss dict in train mode.
  We call model.train() solely to access the loss dict, not to affect
  BatchNorm/Dropout statistics — torch.no_grad() prevents weight updates.
  This is the standard workaround for torchvision detection models.

TODO (future):
  - AMP (Automatic Mixed Precision): add torch.autocast + GradScaler
    when cfg.training.amp is True. Insertion point marked below.
  - Gradient accumulation: batch effective size =
    batch_size * accumulation_steps. Insertion point marked below.
  - ReduceLROnPlateau: scheduler.step(metric) already supported
    via _scheduler_step() helper below.
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
from objdet.utils.checkpoint import (
    save_checkpoint, save_best_checkpoint,
    load_checkpoint, cleanup_old_checkpoints,
)


def build_optimizer(model, training_cfg):
    from torch.optim import SGD, Adam, AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    name = training_cfg.optimizer.lower()
    if name == "sgd":
        opt = SGD(params, lr=training_cfg.learning_rate,
                  momentum=training_cfg.momentum,
                  weight_decay=training_cfg.weight_decay)
    elif name == "adam":
        opt = Adam(params, lr=training_cfg.learning_rate,
                   weight_decay=training_cfg.weight_decay)
    elif name == "adamw":
        opt = AdamW(params, lr=training_cfg.learning_rate,
                    weight_decay=training_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    print(
        f"[Optimizer] {name.upper()} | lr={training_cfg.learning_rate} | "
        f"wd={training_cfg.weight_decay} | "
        f"params={sum(p.numel() for p in params):,}"
    )
    return opt


def build_scheduler(optimizer, training_cfg):
    from torch.optim.lr_scheduler import (
        StepLR, CosineAnnealingLR, ConstantLR,
        LinearLR, SequentialLR, ReduceLROnPlateau,
    )
    name = training_cfg.lr_scheduler.lower()

    if name == "step":
        return StepLR(
            optimizer,
            step_size=training_cfg.lr_step_size,
            gamma=training_cfg.lr_gamma,
        )

    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.epochs,
            eta_min=training_cfg.learning_rate * 0.001,  # fixed: was 0.01
        )

    elif name == "cosine_warmup":
        warmup_epochs = getattr(training_cfg, "warmup_epochs", 3)
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.epochs - warmup_epochs,
            eta_min=training_cfg.learning_rate * 0.001,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(training_cfg, "plateau_factor",   0.5),
            patience=getattr(training_cfg, "plateau_patience", 5),
            min_lr=training_cfg.learning_rate * 0.001,
        )

    elif name == "none":
        return ConstantLR(optimizer, factor=1.0, total_iters=0)

    else:
        raise ValueError(f"Unknown scheduler: {name}")
    

class Trainer:
    """
    Training loop with full loss/metric history tracking.

    History structure:
    {
        "train": {
            "epoch":           [1, 2, ...],
            "total_loss":      [...],
            "loss_classifier": [...],
            "loss_box_reg":    [...],
            "loss_objectness": [...],
            "loss_rpn_box_reg":[...],
        },
        "val": {
            "epoch":           [1, 2, ...],   # every validate_every epochs
            "total_loss":      [...],
            "loss_classifier": [...],
            "loss_box_reg":    [...],
            "loss_objectness": [...],
            "loss_rpn_box_reg":[...],
            "map_50_95":       [...],   # mAP@[0.5:0.95]
            "map_50":          [...],   # mAP@0.50
            "map_75":          [...],   # mAP@0.75
            "ap_per_class":    [{...}, ...],
            "precision":       [...],
            "recall":          [...],
        }
    }
    """

    LOSS_KEYS = [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]

    # TensorBoard display tags — pretty names only for UI, not used internally
    _TB_METRIC_TAGS = {
        "map_50_95": "epoch/val_mAP[.5:.95]",
        "map_50":    "epoch/val_mAP@0.50",
        "map_75":    "epoch/val_mAP@0.75",
        "precision": "epoch/val_precision",
        "recall":    "epoch/val_recall",
    }

    # MLflow metric keys — no special chars allowed in MLflow keys
    _MLF_METRIC_KEYS = {
        "map_50_95": "val_mAP_50_95",
        "map_50":    "val_mAP_50",
        "map_75":    "val_mAP_75",
        "precision": "val_precision",
        "recall":    "val_recall",
    }

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

        self.ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── History — uses standardized internal metric keys ──────────
        self.history = {
            "train": {k: [] for k in ["epoch", "total_loss"] + self.LOSS_KEYS},
            "val": {k: [] for k in (
                ["epoch", "total_loss"] + self.LOSS_KEYS +
                ["map_50_95", "map_50", "map_75",   # standardized keys
                 "ap_per_class", "precision", "recall"]
            )},
        }
        self.history_path = self.ckpt_dir / "training_history.json"

        # Best checkpoint tracking
        # Primary  : highest map_50_95
        # Tie-break: lowest loss (when map_50_95 equal to 4dp)
        self.best_map_50_95 = 0.0
        self.best_loss_at_best_map = float("inf")
        # Early stopping
        self.early_stopping = getattr(cfg.training, "early_stopping", False)
        self.es_patience = getattr(cfg.training, "early_stopping_patience", 5)
        self.es_min_delta = getattr(cfg.training, "early_stopping_min_delta", 1e-4)
        self.es_metric = getattr(cfg.training, "early_stopping_metric", "map_50_95")

        self.no_improve_epochs = 0
        self.best_es_metric = -float("inf")

        print(f"[Trainer] Device        : {self.device}")
        print(f"[Trainer] Checkpoints   : {self.ckpt_dir}")
        print(f"[Trainer] History file  : {self.history_path}")
        print(f"[Trainer] Validate every: {cfg.checkpointing.validate_every} epoch(s)")
        print(f"[Trainer] Save every    : {cfg.checkpointing.save_every} epoch(s)")

        if self.tb_logger:
            hparam_dict = {
                "lr":               cfg.training.learning_rate,
                "optimizer":        cfg.training.optimizer,
                "batch_size":       cfg.training.batch_size,
                "epochs":           cfg.training.epochs,
                "backbone":         cfg.model.backbone_weights,
                "trainable_layers": cfg.model.trainable_backbone_layers,
                "loss_cls":         cfg.loss.classification,
                "loss_box":         cfg.loss.box_regression,
                "lr_scheduler":     cfg.training.lr_scheduler,
                "weight_decay":     cfg.training.weight_decay,
            }
            metric_dict = {
                "epoch/val_mAP[.5:.95]": 0.0,
                "epoch/val_mAP@0.50":    0.0,
            }
            self.tb_logger.log_hparams(hparam_dict, metric_dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: str | Path):
        """Load checkpoint and restore history."""
        self.start_epoch = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.global_step = self.start_epoch * len(self.train_loader)

        if self.history_path.exists():        
            with open(self.history_path, "r") as f:
                self.history = json.load(f)

            # backward compatibility for old history files
            val_hist = self.history.setdefault("val", {})

            if "map_50_95" not in val_hist:
                val_hist["map_50_95"] = val_hist.pop("map", [])

            # Restore best map seen so far
            val_maps = val_hist.get("map_50_95", [])
            val_losses = val_hist.get("total_loss", [])


            if val_maps:
                self.best_map_50_95 = max(val_maps)
                # Find loss at that best epoch
                best_idx = val_maps.index(self.best_map_50_95)
                if best_idx < len(val_losses):
                    self.best_loss_at_best_map = val_losses[best_idx]
            print(f"[Trainer] Restored history. Best mAP@[.5:.95] so far: {self.best_map_50_95:.4f}")
        print(f"[Trainer] Resumed from epoch {self.start_epoch}")

    def fit(self):
        """Full training loop."""
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} / {self.cfg.training.epochs}")
            print(f"{'='*60}")

            # ── Train ─────────────────────────────────────────────────
            train_losses = self._train_one_epoch(epoch)

            # NaN guard — catches exploding gradients silently
            if not torch.isfinite(torch.tensor(train_losses["total_loss"])):
                print(
                    f"[Trainer] WARNING: NaN/Inf loss at epoch {epoch+1}. "
                    "Stopping training. Check learning rate and grad_clip."
                )
                break

            self.history["train"]["epoch"].append(epoch + 1)
            self.history["train"]["total_loss"].append(train_losses["total_loss"])
            for k in self.LOSS_KEYS:
                self.history["train"][k].append(train_losses.get(k, 0.0))

            # ── Validate ───────────────────────────────────────────────
            # validate_every and save_every are now independent.
            # Always validate on the last epoch regardless.
            run_val = (
                (epoch + 1) % self.cfg.checkpointing.validate_every == 0
                or (epoch + 1) == self.cfg.training.epochs
            )

            if run_val:
                val_losses, val_metrics = self._validate(epoch)

                # ── Normalize metric keys to internal standard ─────────
                # COCOEvaluator may return "map" for mAP@[.5:.95].
                # We rename here so the rest of the code is consistent.
                val_metrics = _normalize_metric_keys(val_metrics)

                self.history["val"]["epoch"].append(epoch + 1)
                self.history["val"]["total_loss"].append(val_losses["total_loss"])
                for k in self.LOSS_KEYS:
                    self.history["val"][k].append(val_losses.get(k, 0.0))
                self.history["val"]["map_50_95"].append(val_metrics.get("map_50_95", 0.0))
                self.history["val"]["map_50"].append(val_metrics.get("map_50", 0.0))
                self.history["val"]["map_75"].append(val_metrics.get("map_75", 0.0))
                self.history["val"]["ap_per_class"].append(val_metrics.get("ap_per_class", {}))
                self.history["val"]["precision"].append(val_metrics.get("precision", 0.0))
                self.history["val"]["recall"].append(val_metrics.get("recall", 0.0))

            else:
                val_losses  = {"total_loss": float("nan")}
                val_metrics = {"map_50_95": float("nan")}
                    # -------------------------------------------------
            # Early stopping
            # -------------------------------------------------
            if self.early_stopping:

                current_metric = val_metrics.get(self.es_metric, 0.0)

                improved = (
                    current_metric > self.best_es_metric + self.es_min_delta
                )

                if improved:
                    self.best_es_metric = current_metric
                    self.no_improve_epochs = 0

                    print(
                        f"[EarlyStopping] Improvement detected "
                        f"({self.es_metric}={current_metric:.4f})"
                    )

                else:
                    self.no_improve_epochs += 1

                    print(
                        f"[EarlyStopping] No improvement for "
                        f"{self.no_improve_epochs}/{self.es_patience} epochs"
                    )

                    if self.no_improve_epochs >= self.es_patience:
                        print(
                            f"\n[EarlyStopping] Triggered. "
                            f"No improvement in {self.es_patience} epochs."
                        )
                        break


            # ── LR scheduler step ──────────────────────────────────────
            self._scheduler_step(val_metrics.get("map_50_95") if run_val else None)
            lr = self.optimizer.param_groups[0]["lr"]

            # ── Console summary ────────────────────────────────────────
            print(
                f"\n[Epoch {epoch+1}] lr={lr:.6f} | "
                f"train_loss={train_losses['total_loss']:.4f} | "
                f"cls={train_losses.get('loss_classifier', 0):.4f} | "
                f"box={train_losses.get('loss_box_reg', 0):.4f} | "
                f"obj={train_losses.get('loss_objectness', 0):.4f} | "
                f"rpn={train_losses.get('loss_rpn_box_reg', 0):.4f}"
            )
            if run_val:
                print(
                    f"[Epoch {epoch+1}] val_loss={val_losses['total_loss']:.4f} | "
                    f"mAP@[.5:.95]={val_metrics.get('map_50_95', 0):.4f} | "
                    f"mAP@50={val_metrics.get('map_50', 0):.4f} | "
                    f"mAP@75={val_metrics.get('map_75', 0):.4f}"
                )
                if "ap_per_class" in val_metrics:
                    print(f"[Epoch {epoch+1}] Per-class AP:")
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
                    for internal_key, tb_tag in self._TB_METRIC_TAGS.items():
                        if internal_key in val_metrics:
                            self.tb_logger.log_scalar(
                                tb_tag, val_metrics[internal_key], epoch
                            )
                    for cls_name, ap_val in val_metrics.get("ap_per_class", {}).items():
                        self.tb_logger.log_scalar(
                            f"per_class_ap/{cls_name}", ap_val, epoch
                        )

            # ── MLflow ─────────────────────────────────────────────────
            if self.mlf_logger:
                metrics_to_log = {
                    "train_total_loss": train_losses["total_loss"],
                    "lr": lr,
                }
                metrics_to_log.update(
                    {f"train_{k}": train_losses.get(k, 0.0) for k in self.LOSS_KEYS}
                )
                if run_val:
                    metrics_to_log["val_total_loss"] = val_losses["total_loss"]
                    metrics_to_log.update(
                        {f"val_{k}": val_losses.get(k, 0.0) for k in self.LOSS_KEYS}
                    )
                    for internal_key, mlf_key in self._MLF_METRIC_KEYS.items():
                        if internal_key in val_metrics:
                            metrics_to_log[mlf_key] = val_metrics[internal_key]
                    for cls_name, ap_val in val_metrics.get("ap_per_class", {}).items():
                        metrics_to_log[f"ap_{cls_name}"] = ap_val

                self.mlf_logger.log_metrics(metrics_to_log, step=epoch)

            # ── Persist history ────────────────────────────────────────
            self._save_history()

            # ── Checkpoint ────────────────────────────────────────────
            run_save = (
                (epoch + 1) % self.cfg.checkpointing.save_every == 0
                or (epoch + 1) == self.cfg.training.epochs
            )
            if run_save:
                current_map = val_metrics.get("map_50_95", 0.0) if run_val else 0.0
                current_loss = train_losses["total_loss"]

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=current_loss,
                    map_50_95=current_map,
                    experiment_name=self.cfg.experiment_name,
                    save_dir=self.ckpt_dir,
                    extra={"history": self.history},
                )
                cleanup_old_checkpoints(
                    self.ckpt_dir, keep=self.cfg.checkpointing.keep_last
                )

                # ── Best checkpoint: primary=mAP, tie-break=loss ───────
                if run_val and not torch.isnan(torch.tensor(current_map)):
                    is_better_map  = current_map > self.best_map_50_95
                    is_equal_map   = abs(current_map - self.best_map_50_95) < 1e-4
                    is_better_loss = current_loss < self.best_loss_at_best_map

                    if is_better_map or (is_equal_map and is_better_loss):
                        self.best_map_50_95        = current_map
                        self.best_loss_at_best_map = current_loss
                        save_best_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch + 1,
                            loss=current_loss,
                            map_50_95=current_map,
                            experiment_name=self.cfg.experiment_name,
                            save_dir=self.ckpt_dir,
                            extra={"history": self.history},
                        )

        print("\n[Trainer] Training complete.")
        print(
            f"[Trainer] Best mAP@[.5:.95] = {self.best_map_50_95:.4f} "
            f"(loss at best = {self.best_loss_at_best_map:.4f})"
        )
        print(f"[Trainer] History saved to: {self.history_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scheduler_step(self, metric: Optional[float] = None):
        """
        Step the LR scheduler.

        Handles both epoch-based schedulers (StepLR, CosineAnnealingLR)
        and metric-based schedulers (ReduceLROnPlateau).
        ReduceLROnPlateau requires scheduler.step(metric).
        """
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is not None and not torch.isnan(torch.tensor(metric)):
                self.scheduler.step(metric)
            # else: skip step — no valid metric this epoch
        else:
            self.scheduler.step()

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        accum = {k: 0.0 for k in ["total_loss"] + self.LOSS_KEYS}
        log_every = self.cfg.logging.log_every
        t0 = time.time()

        # TODO: AMP — wrap forward+backward with torch.autocast here
        # if self.cfg.training.amp:
        #     scaler = torch.cuda.amp.GradScaler()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images  = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict  = self.model(images, targets)
            total_loss = sum(v for v in loss_dict.values())

            # NaN guard per batch
            if not torch.isfinite(total_loss):
                print(
                    f"[Trainer] WARNING: Non-finite loss at batch {batch_idx}. "
                    f"Skipping update."
                )
                continue

            self.optimizer.zero_grad()
            total_loss.backward()
            # TODO: AMP — scaler.scale(total_loss).backward(); scaler.step(); scaler.update()

            if self.cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )
            self.optimizer.step()

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
                    f"rpn={loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.4f} "
                    f"[{elapsed:.1f}s]"
                )
                if self.tb_logger:
                    self.tb_logger.log_scalar(
                        "batch/total_train_loss", total_loss.item(), self.global_step
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
        Compute validation loss and mAP.

        NOTE on model.train() usage:
          torchvision Faster R-CNN returns a loss dict ONLY in train mode.
          model.train() here is required to access losses, NOT to affect
          BatchNorm/Dropout behavior — torch.no_grad() prevents any updates.
          After loss computation, model is switched to eval mode for mAP.
        """
        self.model.train()
        val_accum = {k: 0.0 for k in ["total_loss"] + self.LOSS_KEYS}

        with torch.no_grad():
            for images, targets in self.val_loader:
                images  = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict  = self.model(images, targets)
                total_loss = sum(v for v in loss_dict.values())
                val_accum["total_loss"] += total_loss.item()
                for k in self.LOSS_KEYS:
                    val_accum[k] += loss_dict.get(k, torch.tensor(0.0)).item()

        n = max(len(self.val_loader), 1)
        val_losses = {k: v / n for k, v in val_accum.items()}

        evaluator = COCOEvaluator(self.device, self.cfg.eval)
        evaluator.evaluate(self.model, self.val_loader)
        val_metrics = evaluator.get_metrics()

        return val_losses, val_metrics

    def _save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)


# ===========================================================================
# MODULE-LEVEL HELPER
# ===========================================================================

def _normalize_metric_keys(metrics: dict) -> dict:
    """
    Normalize COCOEvaluator output to internal standard keys.

    COCOEvaluator may return "map" for mAP@[.5:.95] (torchmetrics default).
    We rename to "map_50_95" so all downstream code uses a single key.

    Input keys handled:
      "map"     → "map_50_95"
      "map_50"  → unchanged
      "map_75"  → unchanged
      all others → unchanged
    """
    normalized = {}
    for k, v in metrics.items():
        if k == "map":
            normalized["map_50_95"] = v   # primary rename
        else:
            normalized[k] = v
    return normalized