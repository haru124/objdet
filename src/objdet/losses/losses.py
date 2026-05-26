"""
losses/losses.py

Pluggable loss functions for Faster R-CNN ROI classification and box regression.

How Faster R-CNN losses work
────────────────────────────
The model internally computes 4 losses:
  loss_classifier     → ROI classification (cross-entropy by default)
  loss_box_reg        → ROI box regression (smooth L1 by default)
  loss_objectness     → RPN objectness (binary cross-entropy, not overrideable here)
  loss_rpn_box_reg    → RPN box regression (smooth L1, not overrideable here)

torchvision computes these losses deep inside RoIHeads.fastrcnn_loss().
To override them without subclassing RoIHeads, we patch the functions
that RoIHeads calls. This keeps the rest of the pipeline intact.

IMPORTANT: RPN losses (objectness + rpn_box_reg) are NOT overridden here.
Overriding RPN losses requires subclassing RegionProposalNetwork, which
is beyond the current scope.

Usage
─────
    from objdet.losses.losses import build_loss_fn, patch_roi_head_losses

    # Once after building model:
    patch_roi_head_losses(model, loss_cfg)

    # Training loop is unchanged — model(images, targets) returns the
    # same loss_dict structure, but with your custom losses inside.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss

from objdet.entity.config_entity import LossConfig


# ===========================================================================
# CLASSIFICATION LOSSES
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) #p_t is the model's predicted probability for the true class.
    cross_entropy(p_t) = -log(p_t)  → focal_loss = alpha * (1 - p_t)^gamma * cross_entropy(p_t)
normally cross entropy = y_true * log(y_pred), but since y_true is 1 for the true class and 0 for others, it simplifies to -log(p_t) where p_t is the predicted probability for the true class. The focal loss then adds a modulating factor (1 - p_t)^gamma to down-weight easy examples and a weighting factor alpha to balance classes.
    binary cross entropy: CE(p_t) = -[y*log(p_t) + (1-y)*log(1-p_t)]
    Motivation: down-weights easy examples (high p_t) so training focuses
    on hard/misclassified samples. Originally from RetinaNet but applicable
    to ROI head classification in Faster R-CNN.

    Args:
        alpha : per-class weight scalar (float) or None
        gamma : focusing parameter (default 2.0)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs : [N, num_classes] — raw logits (NOT softmax)
        targets: [N]              — integer class labels
        """
        # Standard cross-entropy gives log(p_t) per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t = probability assigned to the TRUE class
        p_t = torch.exp(-ce_loss)

        # Focal weighting: easy examples (p_t → 1) get multiplied by ~0
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def cross_entropy_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard softmax cross-entropy. torchvision's default."""
    return F.cross_entropy(class_logits, labels)


def focal_loss_fn(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Stateless focal loss wrapper (no nn.Module needed)."""
    loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
    return loss(class_logits, labels)


# ===========================================================================
# BOX REGRESSION LOSSES
# ===========================================================================

def smooth_l1_loss(
    pred_deltas: torch.Tensor,
    target_deltas: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss). torchvision's default for box regression.
    For |x| < beta: 0.5 * x^2 / beta
    For |x| >= beta: |x| - 0.5 * beta
    pred_deltas, target_deltas: [N, 4] (dx, dy, dw, dh)
    """
    return F.smooth_l1_loss(pred_deltas, target_deltas, beta=beta)


def l1_loss(
    pred_deltas: torch.Tensor,
    target_deltas: torch.Tensor,
) -> torch.Tensor:
    """Plain L1 loss. More robust to outliers than L2."""
    return F.l1_loss(pred_deltas, target_deltas)


def giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Generalized IoU loss. torchvision provides generalized_box_iou_loss.
    pred_boxes, target_boxes: [N, 4] in xyxy format.

    GIoU = IoU - (C \ U) / C
    where C is the smallest enclosing box of pred and target.
    GIoU ∈ [-1, 1]; loss = 1 - GIoU.

    Advantage over smooth L1: directly optimises the IoU metric,
    handles non-overlapping boxes better than L1/L2.

    NOTE: torchvision's box regression uses encoded deltas (dx,dy,dw,dh),
    not absolute xyxy boxes. To use IoU-based losses we must decode the
    deltas to xyxy first. This is done in the patched loss function below.
    """
    return generalized_box_iou_loss(pred_boxes, target_boxes, reduction="mean")


def diou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Distance IoU loss.
    DIoU = IoU - (d^2 / c^2)
    where d = distance between box centres, c = diagonal of enclosing box.
    Penalises distance between centres, faster convergence than GIoU.
    """
    iou = _box_iou_elementwise(pred_boxes, target_boxes)
    centre_dist_sq = _centre_distance_sq(pred_boxes, target_boxes)
    enclosing_diag_sq = _enclosing_diagonal_sq(pred_boxes, target_boxes)
    diou = iou - centre_dist_sq / (enclosing_diag_sq + 1e-7)
    return (1.0 - diou).mean()


def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Complete IoU loss.
    CIoU = DIoU - alpha_v * v
    where v measures aspect-ratio consistency,
    alpha_v = v / (1 - IoU + v)  (trade-off weight)
    Best overall convergence for bounding box regression.
    """
    iou = _box_iou_elementwise(pred_boxes, target_boxes)
    centre_dist_sq = _centre_distance_sq(pred_boxes, target_boxes)
    enclosing_diag_sq = _enclosing_diagonal_sq(pred_boxes, target_boxes)
    diou_term = iou - centre_dist_sq / (enclosing_diag_sq + 1e-7)

    # Aspect ratio consistency term
    w_p = pred_boxes[:, 2] - pred_boxes[:, 0]
    h_p = pred_boxes[:, 3] - pred_boxes[:, 1]
    w_t = target_boxes[:, 2] - target_boxes[:, 0]
    h_t = target_boxes[:, 3] - target_boxes[:, 1]
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w_t / (h_t + 1e-7)) - torch.atan(w_p / (h_p + 1e-7))
    ) ** 2

    with torch.no_grad():
        alpha_v = v / (1.0 - iou + v + 1e-7)

    ciou = diou_term - alpha_v * v
    return (1.0 - ciou).mean()


# ---------------------------------------------------------------------------
# IoU helper functions
# ---------------------------------------------------------------------------

def _box_iou_elementwise(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU between paired boxes [N,4] xyxy. Returns [N]."""
    inter_x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    inter_y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    inter_x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    inter_y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    return inter / (union + 1e-7)


def _centre_distance_sq(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between box centres. Returns [N]."""
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2
    return (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2


def _enclosing_diagonal_sq(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Squared diagonal of the smallest enclosing box. Returns [N]."""
    enc_x1 = torch.min(boxes_a[:, 0], boxes_b[:, 0])
    enc_y1 = torch.min(boxes_a[:, 1], boxes_b[:, 1])
    enc_x2 = torch.max(boxes_a[:, 2], boxes_b[:, 2])
    enc_y2 = torch.max(boxes_a[:, 3], boxes_b[:, 3])
    return (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2


# ===========================================================================
# FACTORY — build the right loss callable from config
# ===========================================================================

def build_cls_loss_fn(loss_cfg: LossConfig):
    """
    Return a callable: (class_logits[N,C], labels[N]) → scalar loss.
    """
    if loss_cfg.classification == "cross_entropy":
        return cross_entropy_loss

    elif loss_cfg.classification in ("focal", "focal_loss"):
        alpha = loss_cfg.focal_alpha
        gamma = loss_cfg.focal_gamma

        def _focal(class_logits, labels):
            return focal_loss_fn(class_logits, labels, alpha=alpha, gamma=gamma)

        return _focal
    
    elif loss_cfg.classification == "weighted_cross_entropy":
        # Weights: one per class (index 0 = background)
        # Background weight should be low (< 1.0) to avoid penalizing
        # the dominant class too heavily
        weights = getattr(loss_cfg, "cls_weights", None)
        if weights is None:
            raise ValueError(
                "LossConfig.cls_weights must be set for weighted_cross_entropy. "
                "Provide one weight per class including background (index 0)."
            )
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        def _wce(class_logits, labels):
            w = weights_tensor.to(class_logits.device)
            return F.cross_entropy(class_logits, labels, weight=w)

        return _wce

    else:
        raise ValueError(
            f"Unknown classification loss: '{loss_cfg.classification}'. "
            "Choose: 'cross_entropy' | 'focal' | 'weighted_cross_entropy'."
        )



def build_box_loss_fn(loss_cfg: LossConfig):
    """
    Return a callable: (pred_deltas[N,4], target_deltas[N,4]) → scalar loss.

    For IoU-based losses (giou/diou/ciou), the inputs are DECODED boxes
    in xyxy format. The patching mechanism in patch_roi_head_losses()
    handles the decoding step transparently.
    """
    name = loss_cfg.box_regression

    if name == "smooth_l1":
        beta = loss_cfg.smooth_l1_beta

        def _sl1(pred, target):
            return smooth_l1_loss(pred, target, beta=beta)

        return _sl1

    elif name == "l1":
        return l1_loss

    elif name == "giou":
        return giou_loss

    elif name == "diou":
        return diou_loss

    elif name == "ciou":
        return ciou_loss

    else:
        raise ValueError(
            f"Unknown box regression loss: '{name}'. "
            "Choose: 'smooth_l1' | 'l1' | 'giou' | 'diou' | 'ciou'."
        )


# ===========================================================================
# ROI HEAD LOSS PATCHING
# Monkey-patches model.roi_heads to use custom loss functions.
# This avoids subclassing RoIHeads while keeping the rest of the pipeline intact.
# ===========================================================================

# ============================================================
# CUSTOM ROI HEADS — supports IoU-based box regression losses
# ============================================================

from torchvision.models.detection.roi_heads import RoIHeads
from typing import Dict, List, Optional, Tuple

class CustomRoIHeads(RoIHeads):
    """
    Subclass of RoIHeads that properly supports IoU-based box regression.

    For delta-based losses (smooth_l1, l1): identical to original RoIHeads.
    For IoU-based losses (giou, diou, ciou): decodes predicted and target
    deltas to xyxy absolute boxes using the box_coder, then computes IoU loss.

    Why subclass instead of monkey-patching fastrcnn_loss()?
        fastrcnn_loss() only receives encoded deltas — it has no access to
        the proposal boxes needed for decoding. The proposal boxes are only
        available inside RoIHeads.forward(). Subclassing gives us access.
    """

    def __init__(
        self,
        *args,
        cls_loss_fn,
        box_loss_fn,
        uses_iou_loss: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cls_loss_fn = cls_loss_fn
        self._box_loss_fn = box_loss_fn
        self._uses_iou_loss = uses_iou_loss

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor] = {}

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = self._compute_custom_loss(
                class_logits, box_regression,
                labels, regression_targets,
                proposals,        # ← THIS is what the monkey-patch lacked
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg":    loss_box_reg,
            }
        else:
            boxes, scores, class_labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            for i in range(len(boxes)):
                result.append({
                    "boxes":  boxes[i],
                    "labels": class_labels[i],
                    "scores": scores[i],
                })

        return result, losses

    def _compute_custom_loss(
        self,
        class_logits:        torch.Tensor,
        box_regression:      torch.Tensor,
        labels:              List[torch.Tensor],
        regression_targets:  List[torch.Tensor],
        proposals:           List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        labels_flat            = torch.cat(labels, dim=0)
        regression_targets_flat = torch.cat(regression_targets, dim=0)

        # ── Classification loss ────────────────────────────────────────
        classification_loss = self._cls_loss_fn(class_logits, labels_flat)

        # ── Box regression: positive proposals only ────────────────────
        sampled_pos_inds = torch.where(labels_flat > 0)[0]
        if sampled_pos_inds.numel() == 0:
            return classification_loss, box_regression.sum() * 0.0

        num_classes  = box_regression.shape[1] // 4
        labels_pos   = labels_flat[sampled_pos_inds]

        # Select the 4 deltas for each proposal's ground-truth class
        box_regression_pos = box_regression[sampled_pos_inds]              # [N_pos, C*4]
        box_regression_pos = box_regression_pos.reshape(-1, num_classes, 4)[
            torch.arange(len(labels_pos)), labels_pos
        ]                                                                   # [N_pos, 4]

        regression_targets_pos = regression_targets_flat[sampled_pos_inds] # [N_pos, 4]

        if self._uses_iou_loss:
            # Decode encoded deltas → absolute xyxy boxes
            # box_coder.decode(encoded_deltas, reference_boxes) → xyxy
            proposals_flat = torch.cat(proposals, dim=0)
            proposals_pos  = proposals_flat[sampled_pos_inds]              # [N_pos, 4]

            pred_boxes   = self.box_coder.decode(box_regression_pos,      proposals_pos)
            target_boxes = self.box_coder.decode(regression_targets_pos,  proposals_pos)

            # Squeeze [N,1,4] → [N,4] if torchvision returns extra dim
            pred_boxes   = pred_boxes.squeeze(1) if pred_boxes.dim() == 3 else pred_boxes    # [N,4]
            target_boxes = target_boxes.squeeze(1) if target_boxes.dim() == 3 else target_boxes  # [N,4]

            # Clamp to valid range (decoding can produce negative coords)
            pred_boxes   = pred_boxes.clamp(min=0)
            target_boxes = target_boxes.clamp(min=0)

            box_loss = self._box_loss_fn(pred_boxes, target_boxes)
        else:
            box_loss = self._box_loss_fn(box_regression_pos, regression_targets_pos)

        return classification_loss, box_loss


def patch_roi_head_losses(model, loss_cfg: LossConfig):
    """
    Now replaces roi_heads entirely with CustomRoIHeads subclass,
    which has access to proposals inside forward() and properly decodes
    deltas to xyxy boxes for GIoU/DIoU/CIoU computation.

    Patch model.roi_heads.fastrcnn_loss() to use the losses specified in loss_cfg.

    torchvision's RoIHeads.fastrcnn_loss() is a static/class method that
    computes classification and box regression losses. We replace it with
    a closure that calls our custom losses instead.

    Call this ONCE after build_faster_rcnn(), before training starts.

    Example:
        model = build_faster_rcnn(model_cfg)
        patch_roi_head_losses(model, loss_cfg)
        # Training loop is unchanged from here.

    Tensor shapes inside fastrcnn_loss:
        class_logits  : [total_proposals, num_classes]
        box_regression: [total_proposals, num_classes * 4]
        labels        : [total_proposals]   (int64 class indices; 0 = bg)
        regression_targets: [total_proposals, 4]  (encoded deltas)
    """

    cls_loss_fn = build_cls_loss_fn(loss_cfg)
    box_loss_fn = build_box_loss_fn(loss_cfg)
    uses_iou_loss = loss_cfg.box_regression in ("giou", "diou", "ciou")
    
    print(
        f"[Losses] Replacing roi_heads with CustomRoIHeads | "
        f"cls={loss_cfg.classification} | "
        f"box={loss_cfg.box_regression} | "
        f"iou_decode={uses_iou_loss}"
    )

    old = model.roi_heads   # existing RoIHeads — copy all its parameters

    model.roi_heads = CustomRoIHeads(
        # ── Custom loss functions ──────────────────────────────────
        cls_loss_fn   = cls_loss_fn,
        box_loss_fn   = box_loss_fn,
        uses_iou_loss = uses_iou_loss,
        # ── ROI components (unchanged from existing model) ─────────
        box_roi_pool  = old.box_roi_pool,
        box_head      = old.box_head,
        box_predictor = old.box_predictor,
        # ── Matcher thresholds ─────────────────────────────────────
        fg_iou_thresh = old.proposal_matcher.high_threshold,
        bg_iou_thresh = old.proposal_matcher.low_threshold,
        # ── Sampler ────────────────────────────────────────────────
        batch_size_per_image = old.fg_bg_sampler.batch_size_per_image,
        positive_fraction    = old.fg_bg_sampler.positive_fraction,
        # ── Box coder weights ──────────────────────────────────────
        bbox_reg_weights = old.box_coder.weights,
        # ── Inference thresholds ───────────────────────────────────
        score_thresh      = old.score_thresh,
        nms_thresh        = old.nms_thresh,
        detections_per_img = old.detections_per_img,
    )
    print("[Losses] CustomRoIHeads assigned successfully. GIoU/DIoU/CIoU now work.")

