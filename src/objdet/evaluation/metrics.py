"""
evaluation/metrics.py

COCO-style mAP evaluation using torchvision's built-in CocoEvaluator
(wrapped around pycocotools).

We keep this module thin:
  1. Run the model in eval mode over a DataLoader.
  2. Accumulate predictions.
  3. Return a metric dict with mAP@[.5:.95] and mAP@50.

TODO: Add per-class AP breakdown once pycocotools COCO object is available.
TODO: Support segmentation masks if the task expands beyond bounding boxes.
"""

from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert


class COCOEvaluator:
    """
    Lightweight COCO evaluator using pycocotools.

    If pycocotools is not installed, falls back to a dummy evaluator that
    returns zeros — this lets the training loop run without crashing on
    environments that lack pycocotools.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._predictions: list[dict] = []
        self._ground_truths: list[dict] = []

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader):
        """Collect all predictions and ground-truths from *data_loader*."""
        self._predictions.clear()
        self._ground_truths.clear()

        model.eval()
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)   # list of dicts per image

                for target, output in zip(targets, outputs):
                    img_id = target["image_id"].item()

                    self._ground_truths.append({
                        "image_id": img_id,
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    })
                    self._predictions.append({
                        "image_id": img_id,
                        "boxes": output["boxes"].cpu(),
                        "labels": output["labels"].cpu(),
                        "scores": output["scores"].cpu(),
                    })

    def get_metrics(self) -> dict[str, float]:
        """
        Compute COCO mAP metrics.

        Returns a dict with at minimum:
            map      : mAP@[0.50:0.95]
            map_50   : mAP@0.50
            map_75   : mAP@0.75
        """
        try:
            return self._compute_coco_map()
        except Exception as e:
            print(f"[COCOEvaluator] Metric computation failed: {e}")
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_coco_map(self) -> dict[str, float]:
        """
        Use pycocotools for AP computation.

        TODO: Replace this stub with a proper pycocotools COCO object
              constructed from the Cityscapes ground-truth annotations file.
              Currently performs a simplified IoU-threshold sweep.
        """
        # Simplified per-threshold mAP (not full COCO averaging)
        thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 … 0.95
        ap_per_thresh: list[float] = []

        for thresh in thresholds:
            ap = self._mean_ap_at_iou(thresh)
            ap_per_thresh.append(ap)

        map_all = sum(ap_per_thresh) / len(ap_per_thresh)
        map_50 = ap_per_thresh[0]   # threshold 0.50
        map_75 = ap_per_thresh[5]   # threshold 0.75

        return {"map": map_all, "map_50": map_50, "map_75": map_75}

    def _mean_ap_at_iou(self, iou_threshold: float) -> float:
        """
        Compute mean AP over all classes at a fixed IoU threshold.
        This is a basic implementation for educational clarity.
        """
        from collections import defaultdict

        # Collect predictions and GTs per class
        class_preds: dict[int, list] = defaultdict(list)   # label → [(score, tp)]
        class_n_gt: dict[int, int] = defaultdict(int)

        for gt, pred in zip(self._ground_truths, self._predictions):
            gt_boxes = gt["boxes"]      # [M, 4]
            gt_labels = gt["labels"]    # [M]
            pred_boxes = pred["boxes"]  # [K, 4]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            # Count ground-truth per class
            for lbl in gt_labels.tolist():
                class_n_gt[lbl] += 1

            # Sort predictions by descending score
            if len(pred_scores) == 0:
                continue
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]
            pred_scores = pred_scores[order]

            matched_gt = set()
            for pb, pl, ps in zip(pred_boxes, pred_labels.tolist(), pred_scores.tolist()):
                # Find GT boxes with the same label
                same_cls = (gt_labels == pl).nonzero(as_tuple=True)[0]
                best_iou = 0.0
                best_idx = -1

                for gi in same_cls.tolist():
                    if gi in matched_gt:
                        continue
                    iou = _box_iou_single(pb, gt_boxes[gi])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi

                tp = 1 if best_iou >= iou_threshold and best_idx >= 0 else 0
                if tp:
                    matched_gt.add(best_idx)
                class_preds[pl].append((ps, tp))

        # Compute AP per class, then average
        aps: list[float] = []
        for cls, preds_list in class_preds.items():
            n_gt = class_n_gt.get(cls, 0)
            if n_gt == 0:
                continue
            preds_list.sort(key=lambda x: -x[0])
            tp_cum, fp_cum = 0, 0
            precisions, recalls = [], []
            for _, tp in preds_list:
                if tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / n_gt)
            aps.append(_voc_ap(precisions, recalls))

        return sum(aps) / len(aps) if aps else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_iou_single(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    inter_x1 = max(box_a[0].item(), box_b[0].item())
    inter_y1 = max(box_a[1].item(), box_b[1].item())
    inter_x2 = min(box_a[2].item(), box_b[2].item())
    inter_y2 = min(box_a[3].item(), box_b[3].item())

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _voc_ap(precisions: list[float], recalls: list[float]) -> float:
    """11-point VOC interpolated AP."""
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_t) if p_at_t else 0.0
    return ap / 11.0