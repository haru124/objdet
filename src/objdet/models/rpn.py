"""
models/rpn.py

This module does NOT reimplement the RPN.
It wraps torchvision's RegionProposalNetwork with debug utilities.

RPN Architecture recap
──────────────────────
For each FPN level, a shared 3×3 conv (RPNHead) slides over the feature map.
At each spatial location it predicts:
  - objectness score   per anchor → [B, num_anchors, H, W]
  - box delta          per anchor → [B, num_anchors*4, H, W]

Default anchors: 3 scales × 3 aspect ratios = 9 anchors per location
(but torchvision FPN uses 3 anchors per level, not 9).

Specifically for Faster R-CNN + FPN:
  Each FPN level uses 1 scale and 3 aspect ratios → 3 anchors/location.
  Anchor sizes by level: 32², 64², 128², 256², 512² pixels.

RPN output → proposals (after NMS):
  Training: up to 2000 proposals passed to ROI heads
  Inference: up to 1000 proposals passed to ROI heads

Tensor flow
───────────
FPN outputs {P2..P6}: each [B, 256, Hi, Wi]
        ↓
RPNHead (shared 3×3 conv across all levels):
  objectness logits : list of [B, 3, Hi, Wi]   (3 anchors/location)
  bbox deltas       : list of [B, 12, Hi, Wi]   (3 anchors × 4 coords)
        ↓
AnchorGenerator → anchor boxes in image coordinates
        ↓
box_coder.decode(deltas, anchors) → proposal boxes 
        ↓
clip_boxes_to_image + filter_small_boxes + NMS 
#clip_boxes_to_image ensures proposals are within image boundaries.
#filter_small_boxes removes proposals smaller than min_size.
        ↓
proposals: list of [N_i, 4]  (N_i proposals per image, variable)
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from objdet.entity.config_entity import ModelConfig


def get_rpn_from_model(model):
    """
    Extract the RPN sub-module from a built Faster R-CNN model.
    model.rpn is a RegionProposalNetwork.
    """
    return model.rpn


def debug_rpn(image_height: int = 600, image_width: int = 800, batch_size: int = 2):
    """
    Hooks into:
      - model.rpn.head        → captures objectness logits + bbox deltas
                                 per FPN level (real tensors)
      - model.rpn             → captures proposals list after NMS
      - AnchorGenerator       → captures generated anchor counts

    Every shape/count printed comes from real tensors during forward pass.
    """
    from objdet.models.detector import build_faster_rcnn

    print("\n" + "=" * 65)
    print("DEBUG: RPN — real tensor shapes from forward hooks")
    print("=" * 65)

    cfg = ModelConfig(backbone_weights="none", num_classes=9)
    model = build_faster_rcnn(cfg)
    model.eval()

    captured = {}
    hooks = []

    # ------------------------------------------------------------------
    # Hook on RPNHead
    # RPNHead.forward(features) → (objectness_logits, pred_bbox_deltas)
    # features: list of [B, 256, Hi, Wi] — one per FPN level
    # objectness_logits: list of [B, num_anchors, Hi, Wi]
    # pred_bbox_deltas:  list of [B, num_anchors*4, Hi, Wi]
    # ------------------------------------------------------------------
    def rpn_head_hook(module, inp, output):
        logits, deltas = output
        # inp[0] is the list of feature maps from FPN
        captured["rpn_head_input_shapes"] = [list(f.shape) for f in inp[0]]
        captured["rpn_objectness_logits"] = [list(t.shape) for t in logits]
        captured["rpn_bbox_deltas"]       = [list(t.shape) for t in deltas]
        captured["n_fpn_levels"]          = len(logits)

    hooks.append(model.rpn.head.register_forward_hook(rpn_head_hook))

    # ------------------------------------------------------------------
    # Hook on AnchorGenerator
    # AnchorGenerator.forward(image_list, feature_maps) → list of anchor tensors
    # One tensor per image, each [total_anchors_across_levels, 4]
    # ------------------------------------------------------------------
    def anchor_hook(module, inp, output):
        # output: list (one per image) of [N_anchors, 4]
        captured["anchors_per_image"]  = [list(a.shape) for a in output]
        captured["total_anchors"]      = [a.shape[0] for a in output]

    hooks.append(model.rpn.anchor_generator.register_forward_hook(anchor_hook))

    # ------------------------------------------------------------------
    # Hook on the full RPN module
    # RPN.forward returns (boxes, losses)
    # In eval mode: boxes = list of [N_proposals, 4] per image
    # ------------------------------------------------------------------
    def rpn_full_hook(module, inp, output):
        boxes, losses = output
        # boxes is a list of tensors, one per image
        captured["rpn_proposals_per_image"] = [list(b.shape) for b in boxes]
        captured["rpn_proposal_counts"]     = [b.shape[0] for b in boxes]

    hooks.append(model.rpn.register_forward_hook(rpn_full_hook))

    # ------------------------------------------------------------------
    # Run real forward pass
    # model() in eval mode runs backbone → RPN → ROI heads
    # All hooks fire during this single call
    # ------------------------------------------------------------------
    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]

    with torch.no_grad():
        predictions = model(dummy_images)

    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Print captured real shapes
    # ------------------------------------------------------------------
    print(f"\nInput: {batch_size} images [{3}, {image_height}, {image_width}]")

    print(f"\nRPN Head — inputs (FPN feature maps, one per level):")
    for i, shape in enumerate(captured.get("rpn_head_input_shapes", [])):
        print(f"  FPN level {i} → {shape}  (anchors_per_loc × 1 scores)")

    print(f"\nRPN Head — objectness logits (one tensor per FPN level):")
    for i, shape in enumerate(captured.get("rpn_objectness_logits", [])):
        # shape: [B, num_anchors_per_location, Hi, Wi]
        # num_anchors_per_location = 3 for FPN (3 aspect ratios, 1 scale per level)
        n_anchors = shape[1]
        hi, wi    = shape[2], shape[3]
        total_at_level = n_anchors * hi * wi
        print(f"  Level {i}: {shape}  "
              f"({n_anchors} anchors/loc × {hi}×{wi} locs = {total_at_level} anchors)")

    print(f"\nRPN Head — bbox deltas (one tensor per FPN level):")
    for i, shape in enumerate(captured.get("rpn_bbox_deltas", [])):
        # shape: [B, num_anchors*4, Hi, Wi]
        print(f"  Level {i}: {shape}  ({shape[1]//4} anchors × 4 coords)")

    print(f"\nAnchor Generator output:")
    for i, (shape, total) in enumerate(zip(
        captured.get("anchors_per_image", []),
        captured.get("total_anchors", []),
    )):
        print(f"  Image {i}: {shape}  ({total:,} total anchors across all FPN levels)")

    print(f"\nRPN proposals after NMS (eval: up to 1000/image, train: up to 2000/image):")
    for i, (shape, count) in enumerate(zip(
        captured.get("rpn_proposals_per_image", []),
        captured.get("rpn_proposal_counts", []),
    )):
        print(f"  Image {i}: {shape}  ({count} proposals survived NMS)")

    print(f"\nFinal detections per image (post ROI-heads):")
    for i, pred in enumerate(predictions):
        print(f"  Image {i}: {pred['boxes'].shape[0]} detections  "
              f"(scores range [{pred['scores'].min():.3f}, {pred['scores'].max():.3f}])"
              if len(pred['scores']) > 0
              else f"  Image {i}: 0 detections")

    print("=" * 65 + "\n")
    return captured, predictions