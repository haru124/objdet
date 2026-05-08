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


def debug_rpn(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    Run a full forward pass through Faster R-CNN in eval mode and
    intercept the RPN outputs via a forward hook.

    Prints:
    - Objectness logits shape at each FPN level
    - Box delta shape at each FPN level
    - Number of proposals after NMS (per image)

    Note: We use eval mode + no_grad so no loss is computed.
    The hook captures intermediate RPN outputs that are not
    returned by the public model() call.
    """
    from objdet.models.detector import build_faster_rcnn

    print("\n" + "="*60)
    print("DEBUG: RPN tensor flow")
    print("="*60)

    cfg = ModelConfig(backbone_weights="none", num_classes=9)
    model = build_faster_rcnn(cfg)
    model.eval()

    # ----------------------------------------------------------------
    # Register a forward hook on the RPN head to capture its raw outputs
    # before NMS. The hook stores the objectness logits and bbox deltas.
    # ----------------------------------------------------------------
    rpn_hook_data = {}

    def _rpn_head_hook(module, inputs, outputs):
        # RPNHead.forward returns (logits, bbox_deltas)
        # logits     : list of [B, num_anchors, Hi, Wi]
        # bbox_deltas: list of [B, num_anchors*4, Hi, Wi]
        logits, deltas = outputs
        rpn_hook_data["objectness_logits"] = logits
        rpn_hook_data["bbox_deltas"] = deltas

    hook = model.rpn.head.register_forward_hook(_rpn_head_hook)

    # Build dummy image list (Faster R-CNN expects a list of [C,H,W] tensors)
    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]
    print(f"\nInput: {batch_size} images of shape [3, {image_height}, {image_width}]")

    with torch.no_grad():
        # In eval mode, model() returns predictions (boxes, labels, scores)
        # The RPN proposals flow internally to ROI heads.
        predictions = model(dummy_images)

    hook.remove()

    # RPN head outputs (one tensor per FPN level)
    print("\nRPN Head outputs per FPN level:")
    for i, (logits, deltas) in enumerate(zip(
        rpn_hook_data["objectness_logits"],
        rpn_hook_data["bbox_deltas"],
    )):
        print(f"  FPN level {i}:")
        print(f"    objectness logits : {list(logits.shape)}")
        #   [B, 3, Hi, Wi]  — 3 anchors per location, 1 score each
        print(f"    bbox deltas       : {list(deltas.shape)}")
        #   [B, 12, Hi, Wi] — 3 anchors × 4 (dx,dy,dw,dh) deltas

    print(f"\nProposals after NMS (eval mode, per image):")
    # In eval mode, predictions contain final detections (post-ROI-heads).
    # To see raw RPN proposals we'd need another hook on rpn.forward.
    # Here we report final detection counts as a proxy.
    for i, pred in enumerate(predictions):
        print(f"  Image {i}: {len(pred['boxes'])} final detections "
              f"(post ROI-heads, score > 0.05)")

    print("="*60 + "\n")
    return rpn_hook_data, predictions