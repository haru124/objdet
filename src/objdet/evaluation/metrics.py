"""
evaluation/metrics.py

Computes:
  - mAP@[0.5:0.95], mAP@0.5, mAP@0.75
  - Per-class AP
  - Precision and Recall at each IoU threshold
"""

import torch
from torch.utils.data import DataLoader
from objdet.entity.config_entity import EvalConfig
from objdet.constants import CITYSCAPES_CLASSES


class COCOEvaluator:
    def __init__(self, device: torch.device, eval_cfg: EvalConfig = None):
        self.device = device
        self.eval_cfg = eval_cfg if eval_cfg is not None else EvalConfig()
        self._predictions = []
        self._ground_truths = []

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader):
        """Collect predictions and ground truths in eval mode."""
        self._predictions.clear()
        self._ground_truths.clear()
        model.eval()

        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)

                for target, output in zip(targets, outputs):
                    img_id = target["image_id"].item()
                    keep = output["scores"] >= self.eval_cfg.score_threshold # Filter predictions by score threshold
                    keep_idx = keep.nonzero(as_tuple=True)[0][
                        : self.eval_cfg.max_detections
                    ] # Keep only top-k detections after thresholding

                    self._ground_truths.append({
                        "image_id": img_id,
                        "boxes":    target["boxes"].cpu(),
                        "labels":   target["labels"].cpu(),
                    })
                    self._predictions.append({
                        "image_id": img_id,
                        "boxes":    output["boxes"][keep_idx].cpu(),
                        "labels":   output["labels"][keep_idx].cpu(),
                        "scores":   output["scores"][keep_idx].cpu(),
                    })
                    #print("GT images:", len(self._ground_truths))
                    #print("Pred images:", len(self._predictions))
                    #print("Sample GT image_ids:", [g["image_id"] for g in self._ground_truths[:5]])
                    #print("Sample Pred image_ids:", [p["image_id"] for p in self._predictions[:5]])

    def get_metrics(self) -> dict:
        """
        Returns:
        {
            "map":          float   mAP@[0.5:0.95]
            "map_50":       float   mAP@0.5
            "map_75":       float   mAP@0.75
            "ap_per_class": {class_name: float}   AP@0.5 per class
            "precision":    float   mean precision @ IoU=0.5
            "recall":       float   mean recall @ IoU=0.5
        }
        """
        try:
            return self._compute_all_metrics()
        except Exception as e:
            print(f"[COCOEvaluator] Metric computation failed: {e}")
            return {
                "map": 0.0, "map_50": 0.0, "map_75": 0.0,
                "ap_per_class": {}, "precision": 0.0, "recall": 0.0,
            }

    def _compute_all_metrics(self) -> dict:
        thresholds = self.eval_cfg.iou_thresholds
        ap_per_thresh = [self._mean_ap_at_iou(t) for t in thresholds]

        def _at(target_t):
            for t, ap in zip(thresholds, ap_per_thresh):
                if abs(t - target_t) < 1e-4:
                    return ap
            return 0.0

        # Per-class AP at IoU=0.5
        ap_per_class = self._ap_per_class_at_iou(0.5)

        # Mean precision + recall at IoU=0.5
        precision, recall = self._mean_precision_recall_at_iou(0.5)

        return {
            "map":          sum(ap_per_thresh) / len(ap_per_thresh),
            "map_50":       _at(0.50),
            "map_75":       _at(0.75),
            "ap_per_class": ap_per_class,
            "precision":    precision,
            "recall":       recall,
        }

    def _mean_ap_at_iou(self, iou_threshold: float) -> float:
        """mAP across all classes at a single IoU threshold."""
        from collections import defaultdict

        class_preds: dict = defaultdict(list) # {class_idx: [(score, is_tp), ...]} for all predictions of that class across dataset
        class_n_gt:  dict = defaultdict(int)  # {class_idx: count} number of GT instances of that class across dataset

        for gt, pred in zip(self._ground_truths, self._predictions):
            #zip together GT and predictions for each image, then iterate over predictions sorted by score 
            # to determine TP/FP based on IoU with unmatched GT boxes of the same class.
            # Accumulate (score, is_tp) for each predicted box in class_preds, and count total GT instances per class in class_n_gt.
            # Finally compute AP per class and average
            for lbl in gt["labels"].tolist(): 
                class_n_gt[lbl] += 1  #all gts grouped by labels in the image

            if len(pred["scores"]) == 0: # No predictions on this image → all GT are false negatives
                continue

            order = torch.argsort(pred["scores"], descending=True) # Sort predictions by confidence score (descending)
            pred_boxes  = pred["boxes"][order] # Reorder predicted boxes, labels, scores according to sorted order
            pred_labels = pred["labels"][order]
            pred_scores = pred["scores"][order]

            matched_gt = set()
            for pb, pl, ps in zip(
                pred_boxes, pred_labels.tolist(), pred_scores.tolist()
            ):
                same_cls = (gt["labels"] == pl).nonzero(as_tuple=True)[0] # Find GT boxes of the same class as the current prediction
                #0 means we want the indices of GT boxes where the label matches the predicted label pl. This gives us candidate GT boxes to compare against for IoU.
                #same_cls looks like tensor([2, 5]) meaning GT boxes at indices 2 and 5 have the same class as the prediction.
                best_iou, best_idx = 0.0, -1
                for gi in same_cls.tolist():
                    if gi in matched_gt:
                        continue
                    iou = _box_iou_single(pb, gt["boxes"][gi])
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                tp = 1 if best_iou >= iou_threshold and best_idx >= 0 else 0
                if tp:
                    matched_gt.add(best_idx)
                class_preds[pl].append((ps, tp))

        aps = []
        for cls, preds_list in class_preds.items():
            n_gt = class_n_gt.get(cls, 0)
            if n_gt == 0:
                continue
            preds_list.sort(key=lambda x: -x[0])
            tp_cum = fp_cum = 0
            precs, recs = [], []
            for _, tp in preds_list:
                tp_cum += tp; fp_cum += (1 - tp)
                precs.append(tp_cum / (tp_cum + fp_cum))
                recs.append(tp_cum / n_gt)
            aps.append(_voc_ap(precs, recs))

        return sum(aps) / len(aps) if aps else 0.0

    def _ap_per_class_at_iou(self, iou_threshold: float) -> dict:
        """
        Returns {class_name: AP_float} for each class present in ground truth.
        Uses CITYSCAPES_CLASSES for name lookup.
        """
        from collections import defaultdict

        class_preds: dict = defaultdict(list)
        class_n_gt:  dict = defaultdict(int)

        for gt, pred in zip(self._ground_truths, self._predictions):
            for lbl in gt["labels"].tolist():
                class_n_gt[lbl] += 1

            if len(pred["scores"]) == 0:
                continue

            order = torch.argsort(pred["scores"], descending=True)
            pred_boxes  = pred["boxes"][order]
            pred_labels = pred["labels"][order]
            pred_scores = pred["scores"][order]

            matched_gt = set()
            for pb, pl, ps in zip(
                pred_boxes, pred_labels.tolist(), pred_scores.tolist()
            ):
                same_cls = (gt["labels"] == pl).nonzero(as_tuple=True)[0]
                best_iou, best_idx = 0.0, -1
                for gi in same_cls.tolist():
                    if gi in matched_gt:
                        continue
                    iou = _box_iou_single(pb, gt["boxes"][gi])
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                tp = 1 if best_iou >= iou_threshold and best_idx >= 0 else 0
                if tp:
                    matched_gt.add(best_idx)
                class_preds[pl].append((ps, tp))

        ap_per_class = {}
        for cls_idx in sorted(class_n_gt.keys()):
            n_gt = class_n_gt[cls_idx]
            cls_name = (
                CITYSCAPES_CLASSES[cls_idx]
                if cls_idx < len(CITYSCAPES_CLASSES)
                else str(cls_idx)
            )
            preds_list = class_preds.get(cls_idx, [])
            if n_gt == 0 or not preds_list:
                ap_per_class[cls_name] = 0.0
                continue
            preds_list.sort(key=lambda x: -x[0])
            tp_cum = fp_cum = 0
            precs, recs = [], []
            for _, tp in preds_list:
                tp_cum += tp; fp_cum += (1 - tp)
                precs.append(tp_cum / (tp_cum + fp_cum))
                recs.append(tp_cum / n_gt)
            ap_per_class[cls_name] = _voc_ap(precs, recs)

        return ap_per_class

    def _mean_precision_recall_at_iou(
        self, iou_threshold: float
    ) -> tuple[float, float]:
        """
        Compute overall precision and recall at a fixed IoU threshold
        by treating all classes together.
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        """
        tp_total = fp_total = fn_total = 0

        for gt, pred in zip(self._ground_truths, self._predictions):
            gt_boxes  = gt["boxes"]
            gt_labels = gt["labels"]
            pred_boxes  = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            if len(pred_scores) == 0:
                fn_total += len(gt_labels)
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes  = pred_boxes[order]
            pred_labels = pred_labels[order]

            matched_gt = set()
            for pb, pl in zip(pred_boxes, pred_labels.tolist()):
                same_cls = (gt_labels == pl).nonzero(as_tuple=True)[0]
                best_iou, best_idx = 0.0, -1
                for gi in same_cls.tolist():
                    if gi in matched_gt:
                        continue
                    iou = _box_iou_single(pb, gt_boxes[gi])
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_iou >= iou_threshold and best_idx >= 0:
                    tp_total += 1
                    matched_gt.add(best_idx)
                else:
                    fp_total += 1

            fn_total += len(gt_labels) - len(matched_gt)

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall    = tp_total / (tp_total + fn_total + 1e-8)
        return precision, recall


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _box_iou_single(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    inter_x1 = max(box_a[0].item(), box_b[0].item())
    inter_y1 = max(box_a[1].item(), box_b[1].item())
    inter_x2 = min(box_a[2].item(), box_b[2].item())
    inter_y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(precisions: list, recalls: list) -> float: 
    """11-point VOC interpolated AP."""
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_t) if p_at_t else 0.0
    return ap / 11.0