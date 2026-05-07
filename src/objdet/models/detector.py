"""
models/detector.py

Builds a Faster R-CNN detector with a ResNet-50 FPN backbone.

We use the high-level torchvision factory `fasterrcnn_resnet50_fpn` and then
replace the box predictor head to match the desired number of classes.

This is the recommended approach from the torchvision docs:
  https://pytorch.org/vision/stable/models/faster_rcnn.html
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from ods.entity.config_entity import ModelConfig


def build_faster_rcnn(model_cfg: ModelConfig) -> FasterRCNN:
    """
    Build and return a Faster R-CNN model configured for *num_classes*.

    Steps:
      1. Load the pretrained model (backbone + RPN + RoI head).
      2. Replace the box classifier head (FastRCNNPredictor) so the output
         dimension matches our class count.

    Args:
        model_cfg: ModelConfig with num_classes and pretrained flags.

    Returns:
        A FasterRCNN nn.Module ready for training or inference.
    """
    # Step 1 — load official pretrained model
    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if model_cfg.pretrained_backbone
        else None
    )

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        # These control the built-in image resize/normalisation transform
        min_size=model_cfg.min_size,
        max_size=model_cfg.max_size,
        trainable_backbone_layers=model_cfg.trainable_backbone_layers,
    )

    # Step 2 — swap the box predictor head
    # roi_heads.box_predictor is a FastRCNNPredictor(in_features, 91)
    # We replace it with one sized for our dataset.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, model_cfg.num_classes
    )

    return model


def get_model_on_device(model_cfg: ModelConfig, device: torch.device) -> FasterRCNN:
    """Build the model and move it to the target device."""
    model = build_faster_rcnn(model_cfg)
    model.to(device)
    return model