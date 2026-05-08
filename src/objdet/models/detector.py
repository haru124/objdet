"""
models/detector.py

Builds the full Faster R-CNN detector using torchvision's factory function.
The box predictor head is always replaced to match num_classes.

Weight loading is handled via backbone.py's build_backbone().
The detector assembles: backbone → RPN → ROI heads using torchvision internals.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from objdet.entity.config_entity import ModelConfig
from objdet.models.backbone import build_backbone


def build_faster_rcnn(model_cfg: ModelConfig) -> FasterRCNN:
    """
    Build a Faster R-CNN model with:
      - backbone loaded per model_cfg.backbone_weights strategy
      - RPN and ROI heads from torchvision (unchanged)
      - box predictor replaced to match model_cfg.num_classes

    Strategy for assembling:
      1. Build backbone (with weights) via backbone.py
      2. Use fasterrcnn_resnet50_fpn(weights=None, backbone=...) to build the
         full model shell with our custom backbone
      3. Swap the box predictor head for the correct num_classes

    Note on torchvision API:
      fasterrcnn_resnet50_fpn() accepts a backbone kwarg in recent torchvision.
      If your version doesn't support it, we use the weights=None path and
      manually replace the backbone after construction.
    """

    # Step 1 — Build backbone with desired weights
    backbone = build_backbone(model_cfg)

    # Step 2 — Build full model shell around our backbone
    # We pass weights=None because our backbone already has weights loaded.
    # min_size/max_size control GeneralizedRCNNTransform (image resize).
    try:
        # Newer torchvision (>=0.13) supports passing backbone directly
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            backbone=backbone,
            num_classes=91,          # temporarily 91 (COCO), replaced below
            min_size=model_cfg.min_size,
            max_size=model_cfg.max_size,
        )
    except TypeError:
        # Older torchvision: build with None weights, then replace backbone
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=model_cfg.min_size,
            max_size=model_cfg.max_size,
            trainable_backbone_layers=model_cfg.trainable_backbone_layers,
        )
        model.backbone = backbone
        print("[Detector] Replaced model.backbone with custom-loaded backbone.")

    # Step 3 — Replace box predictor head
    # roi_heads.box_predictor is FastRCNNPredictor(in_features=1024, num_classes=91)
    # We replace it with one sized for our dataset (e.g. 9 for Cityscapes).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, model_cfg.num_classes
    )
    print(
        f"[Detector] Box predictor replaced: "
        f"in_features={in_features}, num_classes={model_cfg.num_classes}"
    )

    return model


def get_model_on_device(model_cfg: ModelConfig, device: torch.device) -> FasterRCNN:
    """Build the model and move it to the target device."""
    model = build_faster_rcnn(model_cfg)
    model.to(device)
    return model


def debug_detector(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    End-to-end forward pass debug.
    Runs the full model on dummy data and prints shapes at every stage
    by calling debug_backbone, debug_fpn, debug_rpn, debug_roi_heads.
    No training, no dataset required.
    """
    from objdet.models.backbone import debug_backbone
    from objdet.models.fpn import debug_fpn
    from objdet.models.rpn import debug_rpn
    from objdet.models.roi_heads import debug_roi_heads

    h, w, b = image_height, image_width, batch_size

    print("\n" + "#"*60)
    print("# FULL DETECTOR DEBUG — tensor flow at each stage")
    print("#"*60)

    debug_backbone(h, w, b)
    debug_fpn(h, w, b)
    debug_rpn(h, w, b)
    debug_roi_heads(h, w, b)

    print("#"*60)
    print("# FULL DETECTOR DEBUG COMPLETE")
    print("#"*60 + "\n")