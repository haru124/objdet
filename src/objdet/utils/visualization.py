"""
utils/visualization.py

Cityscapes class colours mapped to official dataset RGB values.
Used for bounding box edge colours in both GT and prediction visualisations.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from objdet.constants import CITYSCAPES_CLASSES


# ---------------------------------------------------------------------------
# Official Cityscapes class colours (RGB tuples, 0-255)
# Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# ---------------------------------------------------------------------------
CITYSCAPES_CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "__background__": (0,   0,   0),     # black
    "person":         (220,  20,  60),   # crimson red
    "rider":          (255,   0,   0),   # pure red
    "car":            (0,    0,  142),   # dark blue
    "truck":          (0,    0,   70),   # very dark blue
    "bus":            (0,   60,  100),   # dark teal
    "motorcycle":     (0,    0,  230),   # bright blue
    "bicycle":        (119,  11,  32),   # dark crimson
    "train":          (0,   80,  100),   # dark cyan
}

# Pre-convert to matplotlib hex strings for convenience
def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

CITYSCAPES_CLASS_COLORS_HEX: dict[str, str] = {
    cls: _rgb_to_hex(rgb)
    for cls, rgb in CITYSCAPES_CLASS_COLORS_RGB.items()
}


def _class_color(label_idx: int) -> str:
    """Return the official Cityscapes hex colour for a class index."""
    cls_name = (
        CITYSCAPES_CLASSES[label_idx]
        if label_idx < len(CITYSCAPES_CLASSES)
        else "__background__"
    )
    return CITYSCAPES_CLASS_COLORS_HEX.get(cls_name, "#ffffff")


def _class_name(label_idx: int) -> str:
    return (
        CITYSCAPES_CLASSES[label_idx]
        if label_idx < len(CITYSCAPES_CLASSES)
        else str(label_idx)
    )


def _tensor_to_numpy(image: torch.Tensor | Image.Image) -> np.ndarray:
    """Convert CHW float tensor or PIL image to HWC float numpy [0,1]."""
    if isinstance(image, torch.Tensor):
        return image.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return np.array(image.convert("RGB")) / 255.0


# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------

def draw_boxes(
    image: torch.Tensor | Image.Image,
    boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    score_threshold: float = 0.5,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
    print_coords: bool = False,
):
    """
    Draw bounding boxes on image using official Cityscapes class colours.

    Args:
        image:           CHW float tensor [0,1] or PIL image.
        boxes:           FloatTensor[N, 4] xyxy format.
        labels:          Int64Tensor[N] class indices.
        scores:          FloatTensor[N] confidence scores.
        score_threshold: Skip boxes below this score.
        title:           Plot title.
        save_path:       Save figure here if provided.
        show:            Call plt.show() if True.
        print_coords:    Print box coordinates + labels to stdout.
    """
    img_np = _tensor_to_numpy(image)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)

    if boxes is None or len(boxes) == 0:
        ax.set_title(title + " (no boxes)")
        _finalize(fig, save_path, show)
        return

    if print_coords:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
        print(f"  {'#':<4} {'Class':<15} {'xmin':>6} {'ymin':>6} {'xmax':>6} {'ymax':>6}"
              + ("  Score" if scores is not None else ""))

    for i, box in enumerate(boxes):
        score = scores[i].item() if scores is not None else None
        if score is not None and score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = [round(v, 1) for v in box.tolist()]
        w, h = xmax - xmin, ymax - ymin
        label_idx = labels[i].item() if labels is not None else 0
        label_name = _class_name(label_idx)
        color = _class_color(label_idx)

        rect = patches.Rectangle(
            (xmin, ymin), w, h,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        caption = label_name + (f" {score:.2f}" if score is not None else "")
        ax.text(
            xmin, max(ymin - 2, 0), caption,
            color="white", fontsize=8,
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
        )

        if print_coords:
            score_str = f"  {score:.3f}" if score is not None else ""
            print(f"  {i:<4} {label_name:<15} {xmin:>6} {ymin:>6} {xmax:>6} {ymax:>6}{score_str}")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    _finalize(fig, save_path, show)


# ---------------------------------------------------------------------------
# Side-by-side GT vs Prediction
# ---------------------------------------------------------------------------

def draw_predictions_vs_gt(
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float = 0.5,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
    print_coords: bool = True,
):
    """
    Side-by-side: left = ground truth boxes, right = predicted boxes.
    Both use official Cityscapes class colours.
    Optionally prints coordinates for both GT and predictions to stdout.
    """
    img_np = _tensor_to_numpy(image)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        (axes[0], gt_boxes,   gt_labels,   None,        "Ground Truth"),
        (axes[1], pred_boxes, pred_labels, pred_scores, "Predictions"),
    ]

    for ax, boxes, labels, scores, panel_title in panels:
        ax.imshow(img_np)
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.axis("off")

        if print_coords:
            print(f"\n{'─'*60}")
            print(f"  {panel_title}")
            print(f"{'─'*60}")
            hdr = f"  {'#':<4} {'Class':<15} {'xmin':>7} {'ymin':>7} {'xmax':>7} {'ymax':>7}"
            if scores is not None:
                hdr += "    Score"
            print(hdr)

        if boxes is None or len(boxes) == 0:
            ax.text(0.5, 0.5, "No boxes", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            continue

        for i, box in enumerate(boxes):
            score = scores[i].item() if scores is not None else None
            if score is not None and score < score_threshold:
                continue

            xmin, ymin, xmax, ymax = [round(v, 1) for v in box.tolist()]
            label_idx = labels[i].item() if labels is not None else 0
            label_name = _class_name(label_idx)
            color = _class_color(label_idx)

            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

            caption = label_name + (f" {score:.2f}" if score is not None else "")
            ax.text(
                xmin, max(ymin - 2, 0), caption,
                color="white", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
            )

            if print_coords:
                score_str = f"    {score:.3f}" if score is not None else ""
                print(f"  {i:<4} {label_name:<15} {xmin:>7} {ymin:>7} "
                      f"{xmax:>7} {ymax:>7}{score_str}")

    plt.tight_layout()
    _finalize(fig, save_path, show)


# ---------------------------------------------------------------------------
# Legend helper — show all class colours
# ---------------------------------------------------------------------------

def draw_class_legend(save_path: Optional[Path] = None, show: bool = True):
    """Draw a colour legend for all 8 Cityscapes detection classes."""
    classes = [c for c in CITYSCAPES_CLASSES if c != "__background__"]
    fig, ax = plt.subplots(figsize=(4, len(classes) * 0.5 + 0.5))
    ax.axis("off")
    for i, cls_name in enumerate(classes):
        color = CITYSCAPES_CLASS_COLORS_HEX.get(cls_name, "#ffffff")
        ax.add_patch(patches.Rectangle((0, i), 0.5, 0.8, color=color))
        ax.text(0.6, i + 0.35, cls_name, va="center", fontsize=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.2, len(classes))
    ax.set_title("Cityscapes Detection Classes", fontsize=11, fontweight="bold")
    _finalize(fig, save_path, show)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _finalize(fig, save_path: Optional[Path], show: bool):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[Viz] Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)