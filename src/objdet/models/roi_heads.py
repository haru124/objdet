"""
models/roi_heads.py

This module does NOT reimplement ROI heads.
It wraps torchvision's RoIHeads with debug utilities.

ROI Heads Architecture recap
─────────────────────────────
Input:  proposals from RPN  → list of [N_i, 4] boxes per image
        FPN feature maps    → OrderedDict of [B, 256, Hi, Wi]

Step 1 — ROI Align (MultiScaleRoIAlign)
  For each proposal, extract a 7×7 feature from the appropriate FPN level.
  FPN level assignment: level = floor(4 + log2(sqrt(wh) / 224))
  Clamped to levels [2, 5].
  Output: [total_proposals, 256, 7, 7]

Step 2 — Box Head (TwoMLPHead)
  Flatten → FC(256×7×7 → 1024) → ReLU → FC(1024 → 1024) → ReLU
  Output: [total_proposals, 1024]

Step 3 — Box Predictor (FastRCNNPredictor)
  class scores : FC(1024 → num_classes)     → [total_proposals, num_classes]
  box deltas   : FC(1024 → num_classes × 4) → [total_proposals, num_classes*4]

Step 4 — Post-processing (inference only)
  box_coder.decode(deltas, proposals) → final boxes
  softmax(class_scores) → class probabilities
  NMS per class → final detections

Tensor flow summary
───────────────────
proposals [N_i, 4]  +  FPN features {P2..P5}
         ↓
MultiScaleRoIAlign  →  [ΣN_i, 256, 7, 7]
         ↓
TwoMLPHead          →  [ΣN_i, 1024]
         ↓
FastRCNNPredictor   →  class_logits[ΣN_i, C],  box_deltas[ΣN_i, C×4]
         ↓
(inference) decode + NMS → boxes[K,4], labels[K], scores[K]
"""

import torch
from objdet.entity.config_entity import ModelConfig


def get_roi_heads_from_model(model):
    """Return the RoIHeads sub-module from a Faster R-CNN model."""
    return model.roi_heads


def debug_roi_heads(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    Run a forward pass and intercept the ROI heads intermediate tensors
    via forward hooks to print shapes at every stage.

    Hooks are registered on:
      model.roi_heads.box_roi_pool   → ROI Align output
      model.roi_heads.box_head       → TwoMLPHead output
      model.roi_heads.box_predictor  → final logits + deltas
    """
    from objdet.models.detector import build_faster_rcnn

    print("\n" + "="*60)
    print("DEBUG: ROI Heads tensor flow")
    print("="*60)

    cfg = ModelConfig(backbone_weights="none", num_classes=9)
    model = build_faster_rcnn(cfg)
    model.eval()

    roi_hook_data = {}

    # Hook 1: ROI Align output → [total_proposals, 256, 7, 7]
    def _roi_pool_hook(module, inputs, output):
        roi_hook_data["roi_align_output"] = output

    # Hook 2: TwoMLPHead output → [total_proposals, 1024]
    def _box_head_hook(module, inputs, output):
        roi_hook_data["box_head_output"] = output

    # Hook 3: FastRCNNPredictor output → (class_logits, box_deltas)
    def _box_predictor_hook(module, inputs, outputs):
        class_logits, box_deltas = outputs
        roi_hook_data["class_logits"] = class_logits
        roi_hook_data["box_deltas"] = box_deltas

    h1 = model.roi_heads.box_roi_pool.register_forward_hook(_roi_pool_hook)
    h2 = model.roi_heads.box_head.register_forward_hook(_box_head_hook)
    h3 = model.roi_heads.box_predictor.register_forward_hook(_box_predictor_hook)

    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]
    print(f"\nInput: {batch_size} images [3, {image_height}, {image_width}]")

    with torch.no_grad():
        predictions = model(dummy_images)

    h1.remove(); h2.remove(); h3.remove()

    print("\nROI Align output (MultiScaleRoIAlign):")
    roi_out = roi_hook_data.get("roi_align_output")
    if roi_out is not None:
        print(f"  Shape: {list(roi_out.shape)}")
        print(f"  → [total_proposals_across_batch, 256, 7, 7]")

    print("\nTwoMLPHead output:")
    head_out = roi_hook_data.get("box_head_output")
    if head_out is not None:
        print(f"  Shape: {list(head_out.shape)}")
        print(f"  → [total_proposals, 1024]")

    print("\nFastRCNNPredictor output:")
    cls = roi_hook_data.get("class_logits")
    delta = roi_hook_data.get("box_deltas")
    if cls is not None:
        print(f"  class_logits : {list(cls.shape)}")
        print(f"  → [total_proposals, {cls.shape[1]}]  ({cls.shape[1]} classes)")
    if delta is not None:
        print(f"  box_deltas   : {list(delta.shape)}")
        print(f"  → [total_proposals, num_classes×4]")

    print(f"\nFinal detections per image (post-NMS):")
    for i, pred in enumerate(predictions):
        print(f"  Image {i}: {len(pred['boxes'])} boxes, "
              f"labels={pred['labels'].tolist()[:5]}..., "
              f"scores={[f'{s:.2f}' for s in pred['scores'].tolist()[:5]]}...")

    print("="*60 + "\n")
    return roi_hook_data, predictions