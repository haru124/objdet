"""
evaluation/metrics.py
Now reads IoU thresholds and score_threshold from EvalConfig.
"""

import torch
from torch.utils.data import DataLoader
from objdet.entity.config_entity import EvalConfig


class COCOEvaluator:
    def __init__(self, device: torch.device, eval_cfg: EvalConfig = None):
        self.device = device
        # Use EvalConfig if provided, else fall back to defaults
        if eval_cfg is None:
            eval_cfg = EvalConfig()
        self.eval_cfg = eval_cfg
        self._predictions = []
        self._ground_truths = []

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader):
        self._predictions.clear()
        self._ground_truths.clear()
        model.eval()

        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)

                for target, output in zip(targets, outputs):
                    img_id = target["image_id"].item()
                    # Filter low-confidence predictions per eval config
                    keep = output["scores"] >= self.eval_cfg.score_threshold
                    # Limit max detections per image
                    keep_indices = keep.nonzero(as_tuple=True)[0]
                    keep_indices = keep_indices[:self.eval_cfg.max_detections]

                    self._ground_truths.append({
                        "image_id": img_id,
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    })
                    self._predictions.append({
                        "image_id": img_id,
                        "boxes": output["boxes"][keep_indices].cpu(),
                        "labels": output["labels"][keep_indices].cpu(),
                        "scores": output["scores"][keep_indices].cpu(),
                    })

    def get_metrics(self) -> dict[str, float]:
        try:
            return self._compute_map()
        except Exception as e:
            print(f"[COCOEvaluator] Metric computation failed: {e}")
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

    def _compute_map(self) -> dict[str, float]:
        # Use IoU thresholds from EvalConfig (configurable from YAML)
        thresholds = self.eval_cfg.iou_thresholds
        ap_per_thresh = [self._mean_ap_at_iou(t) for t in thresholds]

        map_all = sum(ap_per_thresh) / len(ap_per_thresh)

        # mAP@50 and mAP@75 — find by threshold value, not index
        def _ap_at(target_thresh):
            for t, ap in zip(thresholds, ap_per_thresh):
                if abs(t - target_thresh) < 1e-4:
                    return ap
            return 0.0

        return {
            "map":    map_all,
            "map_50": _ap_at(0.50),
            "map_75": _ap_at(0.75),
        }

    # _mean_ap_at_iou, _box_iou_single, _voc_ap — unchanged from original
    # (kept here for completeness in your file)
    def _mean_ap_at_iou(self, iou_threshold: float) -> float:
        from collections import defaultdict
        class_preds: dict[int, list] = defaultdict(list)
        class_n_gt: dict[int, int] = defaultdict(int)

        for gt, pred in zip(self._ground_truths, self._predictions):
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            for lbl in gt_labels.tolist():
                class_n_gt[lbl] += 1

            if len(pred_scores) == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]
            pred_scores = pred_scores[order]

            matched_gt = set()
            for pb, pl, ps in zip(pred_boxes, pred_labels.tolist(), pred_scores.tolist()):
                same_cls = (gt_labels == pl).nonzero(as_tuple=True)[0]
                best_iou, best_idx = 0.0, -1
                for gi in same_cls.tolist():
                    if gi in matched_gt:
                        continue
                    iou = _box_iou_single(pb, gt_boxes[gi])
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


def _box_iou_single(box_a, box_b) -> float:
    inter_x1 = max(box_a[0].item(), box_b[0].item())
    inter_y1 = max(box_a[1].item(), box_b[1].item())
    inter_x2 = min(box_a[2].item(), box_b[2].item())
    inter_y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(precisions, recalls) -> float:
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_t) if p_at_t else 0.0
    return ap / 11.0