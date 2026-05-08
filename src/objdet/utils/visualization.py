"""
utils/visualization.py

Helpers to draw ground-truth and predicted bounding boxes on images,
useful for debugging the dataset and reviewing model predictions.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

from objdet.constants import CITYSCAPES_CLASSES


def draw_boxes(
    image: torch.Tensor | Image.Image,
    boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    score_threshold: float = 0.5,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Visualise bounding boxes on an image.

    Args:
        image:           CHW float tensor [0,1] OR PIL image.
        boxes:           FloatTensor[N, 4] in xyxy format.
        labels:          Int64Tensor[N]  — class indices.
        scores:          FloatTensor[N]  — confidence scores (predictions only).
        score_threshold: Skip boxes with score < threshold.
        title:           Plot title.
        save_path:       If provided, save the figure to this path.
        show:            If True, call plt.show().
    """
    # Convert tensor image to HWC numpy for matplotlib
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        import numpy as np
        img_np = np.array(image) / 255.0

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)

    if boxes is None or len(boxes) == 0:
        ax.set_title(title + " (no boxes)")
        _finalize(fig, save_path, show)
        return

    for i, box in enumerate(boxes):
        score = scores[i].item() if scores is not None else None

        # Filter by score threshold when scores are provided
        if score is not None and score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box.tolist()
        w, h = xmax - xmin, ymax - ymin

        label_idx = labels[i].item() if labels is not None else 0
        label_name = (
            CITYSCAPES_CLASSES[label_idx]
            if label_idx < len(CITYSCAPES_CLASSES)
            else str(label_idx)
        )

        color = _class_color(label_idx)
        rect = patches.Rectangle(
            (xmin, ymin), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        caption = label_name
        if score is not None:
            caption += f" {score:.2f}"
        ax.text(
            xmin, max(ymin - 2, 0), caption,
            color="white", fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
        )

    ax.set_title(title)
    ax.axis("off")
    _finalize(fig, save_path, show)


def draw_predictions_vs_gt(
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Side-by-side comparison of ground-truth and predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, boxes, labels, scores, title_str in [
        (axes[0], gt_boxes, gt_labels, None, "Ground Truth"),
        (axes[1], pred_boxes, pred_labels, pred_scores, "Predictions"),
    ]:
        img_np = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title_str)
        ax.axis("off")

        if boxes is None:
            continue
        for i, box in enumerate(boxes):
            score = scores[i].item() if scores is not None else None
            if score is not None and score < score_threshold:
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            label_idx = labels[i].item() if labels is not None else 0
            label_name = (
                CITYSCAPES_CLASSES[label_idx]
                if label_idx < len(CITYSCAPES_CLASSES)
                else str(label_idx)
            )
            color = _class_color(label_idx)
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            caption = label_name + (f" {score:.2f}" if score is not None else "")
            ax.text(xmin, max(ymin - 2, 0), caption, color="white", fontsize=7,
                    bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"))

    plt.tight_layout()
    _finalize(fig, save_path, show)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finalize(fig, save_path: Optional[Path], show: bool):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _class_color(label_idx: int) -> str:
    """Return a distinct colour for each class index."""
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    ]
    return palette[label_idx % len(palette)]