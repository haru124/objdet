"""
inference/inference.py

Test-set evaluation pipeline. Loads a trained checkpoint and:

  1. Computes test loss (all 4 Faster R-CNN loss components)
  2. Computes test mAP, per-class AP, precision, recall
  3. Visualises N sample images side-by-side (GT vs Predictions)
     - Prints predicted + GT bounding box coordinates and labels
  4. Plots training / validation / test loss curves (subplot 1×3, 5 curves each)
  5. Plots mAP metrics for train / val / test
  6. Plots per-class AP for train / val / test
  7. Plots precision and recall for val + test

Usage:
    python -m src.objdet.inference.inference \
        --config  config/config.yaml \
        --exp     config/experiments/exp_01_smoke_test.yaml \
        --ckpt    PROJECT_ROOT/outputs/checkpoints/exp_01_smoke_test/exp_01_smoke_test_epoch_0002_loss_1.2340.pth \
        --n-samples 5 \
        --score-threshold 0.5 \
        --output-dir PROJECT_ROOT/outputs/inference/exp_01_smoke_test
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Make sure src/ is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from objdet.config.configuration import ConfigurationManager
from objdet.datasets.dataloader import build_test_loader,build_dataloader
from objdet.models.detector import get_model_on_device
from objdet.evaluation.metrics import COCOEvaluator
from objdet.utils.checkpoint import load_checkpoint
from objdet.utils.visualization import draw_predictions_vs_gt, draw_class_legend
from objdet.constants import CITYSCAPES_CLASSES
from objdet.tracking.tensorboard_logger import TensorBoardLogger
from objdet.tracking.mlflow_logger import MLflowLogger


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def run_inference(
    config_path: str,
    exp_path: str | None,
    ckpt_path: str,
    n_samples: int = 5,
    score_threshold: float = 0.5,
    output_dir=PROJECT_ROOT / "outputs" / "inference",
    split: str = "test",
    tb_log_dir: str | None = None,
    mlflow_uri: str | None = None,
    sample_seed: int = 0,
    # NEW: accept existing open loggers from main.py when chaining
    existing_mlf_logger=None,
    existing_tb_logger=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    cfg = ConfigurationManager(
        base_config_path=config_path,
        experiment_config_path=exp_path,
    ).get_config()

    # ── Use existing loggers if passed (chained from training) ────────
    # This makes train + test metrics appear in the SAME MLflow run
    # and the SAME TensorBoard run.
    # If running standalone inference, create new loggers.
      # ── Logger ownership ──────────────────────────────────────────────
    # Rule: whoever creates the logger, closes it.
    #
    # Scenario A — chained from training (existing_* is not None):
    #   main.py created the loggers and will close them after this
    #   function returns. We must NOT close them here.
    #
    # Scenario B — standalone inference (existing_* is None):
    #   We create them here, so we must close them here.
    #
    _owns_loggers = existing_mlf_logger is None  # True = we created them, we close them

    if existing_tb_logger is not None:
        tb_logger = existing_tb_logger
    else:
        tb_logger = TensorBoardLogger(
            log_dir=tb_log_dir or cfg.logging.tensorboard_dir,
            experiment_name=cfg.experiment_name,
            run_type="inference",  # separate subfolder from training
        ) if (tb_log_dir or cfg.logging.tensorboard_dir) else None

    if existing_mlf_logger is not None:
        mlf_logger = existing_mlf_logger
    else:
        mlf_logger = MLflowLogger(
            tracking_uri=mlflow_uri or cfg.logging.mlflow_tracking_uri,
            experiment_name=cfg.project_name,
            run_name=cfg.experiment_name,  # same name → resumes or creates
        )

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"\n{'='*65}")
    print(f"  INFERENCE — {cfg.experiment_name}")
    print(f"{'='*65}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*65}\n")

    # ── 2. Model ───────────────────────────────────────────────────────────
    model = get_model_on_device(cfg.model, device)
    load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()

    # ── 3. Test DataLoader ─────────────────────────────────────────────────
    # Cityscapes "test" split has no annotations in the public release.
    # We use "val" as the held-out test set here, which is standard practice.
    # If you have the test annotations, change split="test".
    if split == "val":
        inf_loader = build_dataloader(
            data_cfg = cfg.data,  
            split='val',
            batch_size=1,
            shuffle=False,  # no shuffling for val/test
        )
    else:
        inf_loader = build_test_loader(
            data_cfg=cfg.data,
            batch_size=1,       # batch_size=1 makes sample collection easy
        )
    print(f"[Inference] Test batches: {len(inf_loader)}\n")

    # ── 4. Test Loss ───────────────────────────────────────────────────────
    print("─"*40)
    print("Computing test loss ...")
    print("─"*40)
    inf_losses = _compute_loss(model, inf_loader, device)
    _print_losses("TEST", inf_losses)

    # ── 5. Test mAP + per-class AP ─────────────────────────────────────────
    print("\n" + "─"*40)
    print("Computing test metrics ...")
    print("─"*40)
    evaluator = COCOEvaluator(device, cfg.eval)
    evaluator.evaluate(model, inf_loader)
    inf_metrics = evaluator.get_metrics()
    _print_metrics("TEST", inf_metrics)

    _log_test_results_to_tensorboard(tb_logger, inf_losses, inf_metrics)
    _log_test_results_to_mlflow(mlf_logger, inf_losses, inf_metrics)

    # ── Only close loggers if we created them (standalone inference) ──
    if _owns_loggers:
        if tb_logger:
            tb_logger.close()
        if mlf_logger:
            mlf_logger.end_run()


    # ── 6. Visualize sample images ─────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Visualising {n_samples} test samples ...")
    print(f"{'─'*40}")
    _visualize_samples(
        model=model,
        data_loader=inf_loader,
        device=device,
        n_samples=n_samples,
        score_threshold=score_threshold,
        viz_dir=viz_dir,
        seed=sample_seed,
    )

    # ── 7. Load training history ───────────────────────────────────────────
    # training_history.json is saved by Trainer alongside checkpoints
    history_path = Path(ckpt_path).parent / "training_history.json"
    history = _load_history(history_path)

    # ── 8. Loss plots ──────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Plotting loss curves ...")
    print("─"*40)
    _plot_loss_curves(history, inf_losses, plot_dir)

    # ── 9. mAP + metric plots ──────────────────────────────────────────────
    print("Plotting metric curves ...")
    _plot_map_curves(history, inf_metrics, plot_dir)
    _plot_per_class_ap(history, inf_metrics, plot_dir)
    _plot_precision_recall(history, inf_metrics, plot_dir)

    # ── 10. Save results JSON ──────────────────────────────────────────────
    results = {
        "experiment": cfg.experiment_name,
        "checkpoint": str(ckpt_path),
        "inf_losses": inf_losses,
        "inf_metrics": {
            k: v for k, v in inf_metrics.items()
            if not isinstance(v, dict)
        },
        "test_ap_per_class": inf_metrics.get("ap_per_class", {}),
    }
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Inference] Results saved → {results_path}")
    print("[Inference] Done.\n")


# ===========================================================================
# LOSS COMPUTATION ON TEST SET
# ===========================================================================

LOSS_KEYS = [
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
]


def _compute_loss(model, data_loader, device) -> dict:
    """
    Compute mean loss over the entire data_loader.

    Faster R-CNN only returns a loss dict in TRAIN mode.
    We switch to train mode with no_grad to compute losses
    without updating weights or affecting BatchNorm stats.

    Returns:
        {
          "total_loss":       float,
          "loss_classifier":  float,
          "loss_box_reg":     float,
          "loss_objectness":  float,
          "loss_rpn_box_reg": float,
        }
    """
    model.train()   # must be in train mode to get loss dict
    accum = {k: 0.0 for k in ["total_loss"] + LOSS_KEYS}
    n_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total = sum(v for v in loss_dict.values())

            accum["total_loss"] += total.item()
            for k in LOSS_KEYS:
                accum[k] += loss_dict.get(k, torch.tensor(0.0)).item()
            n_batches += 1

    model.eval()    # switch back to eval mode for prediction
    n = max(n_batches, 1)
    return {k: v / n for k, v in accum.items()}


def _log_test_results_to_tensorboard(tb_logger, inf_losses, inf_metrics):
    if tb_logger is None:
        return
    # Use step=0 — test is a single point, not a timeseries
    tb_logger.log_scalar("test/total_loss",    inf_losses["total_loss"],       0)
    tb_logger.log_scalar("test/loss_cls",      inf_losses["loss_classifier"],  0)
    tb_logger.log_scalar("test/loss_box_reg",  inf_losses["loss_box_reg"],     0)
    tb_logger.log_scalar("test/loss_obj",      inf_losses["loss_objectness"],  0)
    tb_logger.log_scalar("test/loss_rpn_box",  inf_losses["loss_rpn_box_reg"], 0)

    tb_logger.log_scalar("test/mAP[.5:.95]",   inf_metrics["map"],       0)
    tb_logger.log_scalar("test/mAP@0.50",       inf_metrics["map_50"],    0)
    tb_logger.log_scalar("test/mAP@0.75",       inf_metrics["map_75"],    0)
    tb_logger.log_scalar("test/precision",      inf_metrics["precision"], 0)
    tb_logger.log_scalar("test/recall",         inf_metrics["recall"],    0)

    for cls_name, ap_val in inf_metrics.get("ap_per_class", {}).items():
        tb_logger.log_scalar(f"test/ap_{cls_name}", ap_val, 0)

    tb_logger.close()
    print("[TensorBoard] Test results logged.")


def _log_test_results_to_mlflow(mlf_logger, inf_losses, inf_metrics):
    if mlf_logger is None:
        return
    metrics = {
        "test_total_loss":    inf_losses["total_loss"],
        "test_loss_cls":      inf_losses["loss_classifier"],
        "test_loss_box_reg":  inf_losses["loss_box_reg"],
        "test_loss_obj":      inf_losses["loss_objectness"],
        "test_loss_rpn_box":  inf_losses["loss_rpn_box_reg"],
        "test_mAP_5_95":      inf_metrics["map"],
        "test_mAP_50":        inf_metrics["map_50"],
        "test_mAP_75":        inf_metrics["map_75"],
        "test_precision":     inf_metrics["precision"],
        "test_recall":        inf_metrics["recall"],
    }
    for cls_name, ap_val in inf_metrics.get("ap_per_class", {}).items():
        metrics[f"test_ap_{cls_name}"] = ap_val

    mlf_logger.log_metrics(metrics, step=0)
    #mlf_logger.end_run() 
    #NOTE: do NOT call mlf_logger.end_run() here — caller owns lifecycle
    print("[MLflow] Test results logged.")


# ===========================================================================
# SAMPLE VISUALISATION
# ===========================================================================

def _visualize_samples(
    model,
    data_loader,
    device,
    n_samples: int,
    score_threshold: float,
    viz_dir: Path,
    seed: int = 0,
):
    """
    Collect n_samples from data_loader, run inference, and save
    side-by-side GT vs Prediction plots with printed coordinates.
    """
    model.eval()

    if not hasattr(data_loader, "dataset") or not hasattr(data_loader.dataset, "__getitem__"):
        raise ValueError(
            "data_loader.dataset must support indexing for deterministic sample selection"
        )

    dataset = data_loader.dataset
    dataset_length = len(dataset)
    if n_samples > dataset_length:
        raise ValueError(
            f"n_samples={n_samples} is larger than dataset length={dataset_length}"
        )

    indices = np.random.default_rng(seed).choice(dataset_length, size=n_samples, replace=False)

    for sample_idx, idx in enumerate(indices, start=1):
        image, target = dataset[idx]
        images_dev = [image.to(device)]

        with torch.no_grad():
            predictions = model(images_dev)
            pred = predictions[0]

        img_id = target["image_id"].item()
        print(f"\n{'='*65}")
        print(f"  Sample {sample_idx} / {n_samples}   (image_id={img_id})")
        print(f"{'='*65}")

        gt_boxes  = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()
        pred_boxes  = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_scores = pred["scores"].cpu()

        # Filter predictions by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes_filt  = pred_boxes[keep]
        pred_labels_filt = pred_labels[keep]
        pred_scores_filt = pred_scores[keep]

        # Print coordinates
        _print_boxes("GROUND TRUTH", gt_boxes, gt_labels, scores=None)
        _print_boxes(
            "PREDICTIONS (filtered)", pred_boxes_filt,
            pred_labels_filt, pred_scores_filt
        )

        # Save side-by-side figure
        save_path = viz_dir / f"sample_{sample_idx:03d}_imgid_{img_id}.png"
        draw_predictions_vs_gt(
            image=image,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes_filt,
            pred_labels=pred_labels_filt,
            pred_scores=pred_scores_filt,
            score_threshold=0.0,    # already filtered above
            title=f"Sample {sample_idx} | image_id={img_id}",
            save_path=save_path,
            show=False,
            print_coords=False,
        )
        print(f"  [Viz] Saved → {save_path}")


def _print_boxes(title: str, boxes, labels, scores=None):
    """Print bounding box coordinates in a clean table."""
    print(f"\n  {title}  ({len(boxes)} boxes)")
    if len(boxes) == 0:
        print("    (none)")
        return
    hdr = f"    {'#':<4} {'Class':<15} {'xmin':>7} {'ymin':>7} {'xmax':>7} {'ymax':>7}"
    if scores is not None:
        hdr += "    Score"
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [round(v, 1) for v in box.tolist()]
        label_idx = labels[i].item()
        label_name = (
            CITYSCAPES_CLASSES[label_idx]
            if label_idx < len(CITYSCAPES_CLASSES)
            else str(label_idx)
        )
        score_str = f"    {scores[i].item():.3f}" if scores is not None else ""
        print(f"    {i:<4} {label_name:<15} {xmin:>7} {ymin:>7} {xmax:>7} {ymax:>7}{score_str}")


# ===========================================================================
# PLOTTING
# ===========================================================================

def _plot_loss_curves(history: dict, inf_losses: dict, plot_dir: Path):
    """
    3-panel subplot: Training Loss | Validation Loss | Test Loss
    Each panel shows 5 curves: total + 4 component losses.

    Test loss is shown as horizontal dashed lines (single value, not a curve)
    since test is evaluated only once.
    """
    loss_styles = {
        "total_loss":       {"color": "#2c3e50", "lw": 2.5, "label": "Total"},
        "loss_classifier":  {"color": "#e74c3c", "lw": 1.5, "label": "Classifier"},
        "loss_box_reg":     {"color": "#3498db", "lw": 1.5, "label": "Box Reg"},
        "loss_objectness":  {"color": "#2ecc71", "lw": 1.5, "label": "Objectness"},
        "loss_rpn_box_reg": {"color": "#f39c12", "lw": 1.5, "label": "RPN Box Reg"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Loss Curves — Train / Validation / Test", fontsize=14, fontweight="bold")

    # ── Panel 1: Training Loss ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Training Loss", fontsize=12)
    train_epochs = history.get("train", {}).get("epoch", [])
    if train_epochs:
        for key, style in loss_styles.items():
            values = history["train"].get(key, [])
            if values:
                ax.plot(
                    train_epochs, values,
                    color=style["color"], lw=style["lw"], label=style["label"],
                )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Validation Loss ───────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Validation Loss", fontsize=12)
    val_epochs = history.get("val", {}).get("epoch", [])
    if val_epochs:
        for key, style in loss_styles.items():
            values = history["val"].get(key, [])
            if values:
                ax.plot(
                    val_epochs, values,
                    color=style["color"], lw=style["lw"], label=style["label"],
                )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Test Loss (single values → horizontal dashed lines) ────────
    ax = axes[2]
    ax.set_title("Test Loss (final evaluation)", fontsize=12)
    x_range = [0.2, 0.8]   # just a short horizontal span for clarity

    for key, style in loss_styles.items():
        val = inf_losses.get(key, None)
        if val is not None and not np.isnan(val):
            ax.hlines(
                y=val,
                xmin=x_range[0], xmax=x_range[1],
                color=style["color"], lw=style["lw"] + 0.5,
                linestyles="--", label=f"{style['label']}: {val:.4f}",
            )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("Test Set")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = plot_dir / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Loss curves → {out}")


def _plot_map_curves(history: dict, inf_metrics: dict, plot_dir: Path):
    """
    mAP curves for Train (not applicable — no train mAP tracked) and Val,
    plus test mAP as a dashed horizontal marker.

    Since training mAP is not computed (too expensive), we plot:
      - Val mAP, mAP@50, mAP@75 over epochs
      - Test mAP, mAP@50, mAP@75 as horizontal dashed lines
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("mAP Curves — Validation & Test", fontsize=14, fontweight="bold")

    map_styles = {
        "map":    {"color": "#2c3e50", "lw": 2.5, "label": "mAP@[.5:.95]"},
        "map_50": {"color": "#e74c3c", "lw": 1.8, "label": "mAP@0.50"},
        "map_75": {"color": "#3498db", "lw": 1.8, "label": "mAP@0.75"},
    }

    # ── Panel 1: Validation mAP over epochs ───────────────────────────────
    ax = axes[0]
    ax.set_title("Validation mAP", fontsize=12)
    val_epochs = history.get("val", {}).get("epoch", [])
    for key, style in map_styles.items():
        values = history.get("val", {}).get(key, [])
        if values and val_epochs:
            ax.plot(
                val_epochs, values,
                color=style["color"], lw=style["lw"], label=style["label"],
                marker="o", markersize=4,
            )
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 2: Test mAP (bar chart for clarity) ─────────────────────────
    ax = axes[1]
    ax.set_title("Test mAP", fontsize=12)
    metric_names = ["mAP@[.5:.95]", "mAP@0.50", "mAP@0.75"]
    metric_keys  = ["map", "map_50", "map_75"]
    colors       = ["#2c3e50", "#e74c3c", "#3498db"]
    values       = [inf_metrics.get(k, 0.0) for k in metric_keys]

    bars = ax.bar(metric_names, values, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_ylim(0, 1); ax.set_ylabel("mAP"); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = plot_dir / "map_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] mAP curves → {out}")


def _plot_per_class_ap(history: dict, inf_metrics: dict, plot_dir: Path):
    """
    Per-class AP@0.5 grouped bar chart.
    Shows val AP (last epoch) vs test AP for each class.
    """
    classes = [c for c in CITYSCAPES_CLASSES if c != "__background__"]

    # Get last epoch val per-class AP
    val_ap_history = history.get("val", {}).get("ap_per_class", [])
    val_ap = val_ap_history[-1] if val_ap_history else {}
    test_ap = inf_metrics.get("ap_per_class", {})

    val_vals  = [val_ap.get(c, 0.0)  for c in classes]
    test_vals = [test_ap.get(c, 0.0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Per-Class AP@0.5 — Validation vs Test", fontsize=13, fontweight="bold")

    bars_val  = ax.bar(x - width/2, val_vals,  width, label="Validation", color="#3498db", alpha=0.85)
    bars_test = ax.bar(x + width/2, test_vals, width, label="Test",       color="#e74c3c", alpha=0.85)

    # Value labels on bars
    for bar in bars_val:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
    for bar in bars_test:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = plot_dir / "per_class_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Per-class AP → {out}")


def _plot_precision_recall(history: dict, inf_metrics: dict, plot_dir: Path):
    """
    Precision and Recall bar chart for Validation (last epoch) vs Test.
    """
    val_p_history = history.get("val", {}).get("precision", [])
    val_r_history = history.get("val", {}).get("recall", [])
    val_p  = val_p_history[-1]  if val_p_history  else 0.0
    val_r  = val_r_history[-1]  if val_r_history   else 0.0
    test_p = inf_metrics.get("precision", 0.0)
    test_r = inf_metrics.get("recall",    0.0)

    # Also plot val Precision/Recall over epochs if available
    val_epochs = history.get("val", {}).get("epoch", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Precision & Recall — Validation (curve) & Test (bar)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Val Precision + Recall over epochs ────────────────────────
    ax = axes[0]
    ax.set_title("Validation Precision & Recall", fontsize=11)
    if val_epochs and val_p_history:
        ax.plot(val_epochs, val_p_history, color="#3498db", lw=2,
                marker="o", markersize=4, label="Precision")
        ax.plot(val_epochs, val_r_history, color="#e74c3c", lw=2,
                marker="s", markersize=4, label="Recall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 2: Test P + R bar ────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Test Precision & Recall", fontsize=11)
    labels_ = ["Precision", "Recall"]
    values_ = [test_p, test_r]
    colors_ = ["#3498db", "#e74c3c"]
    bars = ax.bar(labels_, values_, color=colors_, alpha=0.85, width=0.4)
    for bar, val in zip(bars, values_):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = plot_dir / "precision_recall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Precision/Recall → {out}")


# ===========================================================================
# UTILITY
# ===========================================================================

def _load_history(path: Path) -> dict:
    """Load training_history.json from trainer output. Returns empty dict if not found."""
    if path.exists():
        with open(path, "r") as f:
            h = json.load(f)
        print(f"[Inference] Loaded training history from {path}")
        return h
    print(f"[Inference] WARNING: training history not found at {path}. "
          "Loss curves will be empty.")
    return {"train": {}, "val": {}}


def _print_losses(split: str, losses: dict):
    print(f"\n  {split} LOSSES")
    print(f"  {'─'*40}")
    print(f"  Total loss        : {losses.get('total_loss', 0):.6f}")
    print(f"  loss_classifier   : {losses.get('loss_classifier', 0):.6f}")
    print(f"  loss_box_reg      : {losses.get('loss_box_reg', 0):.6f}")
    print(f"  loss_objectness   : {losses.get('loss_objectness', 0):.6f}")
    print(f"  loss_rpn_box_reg  : {losses.get('loss_rpn_box_reg', 0):.6f}")


def _print_metrics(split: str, metrics: dict):
    print(f"\n  {split} METRICS")
    print(f"  {'─'*40}")
    print(f"  mAP@[.5:.95]  : {metrics.get('map',    0):.6f}")
    print(f"  mAP@0.50      : {metrics.get('map_50', 0):.6f}")
    print(f"  mAP@0.75      : {metrics.get('map_75', 0):.6f}")
    print(f"  Precision@0.5 : {metrics.get('precision', 0):.6f}")
    print(f"  Recall@0.5    : {metrics.get('recall',    0):.6f}")
    ap_pc = metrics.get("ap_per_class", {})
    if ap_pc:
        print(f"\n  Per-Class AP@0.5:")
        for cls_name, ap in ap_pc.items():
            print(f"    {cls_name:<15}: {ap:.6f}")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Faster R-CNN inference on test set")
    p.add_argument("--config",           default="config/config.yaml")
    p.add_argument("--exp",              default=None)
    p.add_argument("--ckpt",             required=True, help="Path to .pth checkpoint")
    p.add_argument("--n-samples",        type=int,   default=5)
    p.add_argument("--score-threshold",  type=float, default=0.5)
    p.add_argument("--sample-seed",      type=int,   default=0,
                   help="Fixed seed to deterministically choose visualization samples")
    p.add_argument("--output-dir",       default="outputs/inference")
    p.add_argument("--split",            default="test", choices=["val", "test"],
                   help="Which data split to evaluate on (default: test). ")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        config_path=args.config,
        exp_path=args.exp,
        ckpt_path=args.ckpt,
        n_samples=args.n_samples,
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
        split=args.split or "test",
        sample_seed=args.sample_seed,
        
    )