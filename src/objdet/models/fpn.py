"""
models/fpn.py

This module does NOT reimplement FPN.
It wraps torchvision's existing FPN (embedded inside BackboneWithFPN)
with debug utilities to inspect the FPN tensor flow in isolation.

Architecture reminder
─────────────────────
ResNet-50 body produces feature maps at 4 scales (C2–C5).
FPN adds lateral connections + top-down pathway to produce P2–P5
(all with 256 channels), plus P6 via max-pool.

                ┌───────────────────────────────────────┐
  Input image   │   ResNet-50 body (bottom-up pathway)  │
  [B,3,H,W]  → │  C2[/4]  C3[/8]  C4[/16]  C5[/32]    │
                └───────────────────────────────────────┘
                         │       │        │        │
                         ▼       ▼        ▼        ▼
                ┌───────────────────────────────────────┐
                │   FPN (top-down + lateral connections) │
                │  P2[/4] P3[/8] P4[/16] P5[/32] P6[/64]│
                └───────────────────────────────────────┘
                All P levels: 256 channels.

Channel dimensions
──────────────────
C2: 256 ch  → FPN lateral conv (1×1) → 256 ch → P2
C3: 512 ch  → FPN lateral conv (1×1) → 256 ch → P3
C4: 1024 ch → FPN lateral conv (1×1) → 256 ch → P4
C5: 2048 ch → FPN lateral conv (1×1) → 256 ch → P5
P5 max-pool → P6  (used for large-object RPN anchors)
"""

##### ------ IMPORTANT -------######## 
#In torchvision Faster R-CNN, “backbone” means the entire feature extractor stack: ResNet body + FPN together.

#################################################################################################################

import torch
from objdet.entity.config_entity import ModelConfig


def get_fpn_from_backbone(backbone):    #only extracts the already created fpn from backbone. 
    #in case of imagenet weights with backbone only, fpn is randomly initialized and not pretrained, but we can still inspect its tensor flow and output shapes. 
    # In case of loading a full model checkpoint, the fpn weights will be loaded into backbone.fpn, so we can inspect the pretrained fpn tensor flow and output shapes as well.
    """
    Extract the FeaturePyramidNetwork sub-module from a BackboneWithFPN.

    BackboneWithFPN wraps:
        .body  → IntermediateLayerGetter (ResNet-50 body, extracts C2..C5)
        .fpn   → FeaturePyramidNetwork

    Returns the raw FPN module for inspection only.
    Do NOT call fpn() directly during training; use backbone(images) instead.
    """
    return backbone.fpn


def debug_fpn(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    Step through the FPN tensor flow manually:
      1. Run ResNet-50 body → get C2, C3, C4, C5
      2. Run FPN on those → get P2, P3, P4, P5, P6

    Prints channel and spatial dimensions at each stage.

    This makes explicit what the backbone.body and backbone.fpn
    sub-modules do separately, which is hidden when you call backbone(x).
    """
    from objdet.models.backbone import build_backbone

    print("\n" + "="*60)
    print("DEBUG: FPN tensor flow (body → FPN separately), file: fpn.py")
    print("="*60)

    cfg = ModelConfig(backbone_weights="none")
    backbone = build_backbone(cfg)
    backbone.eval()

    dummy_batch = torch.rand(batch_size, 3, image_height, image_width)
    print(f"\nInput batch shape   : {list(dummy_batch.shape)}, file: fpn.py")

    with torch.no_grad():
        # --- Step 1: ResNet body (bottom-up) ---
        # backbone.body is an IntermediateLayerGetter that returns
        # intermediate activations by layer name.
        body_outputs = backbone.body(dummy_batch)  # OrderedDict

        print("\nResNet-50 body outputs (C2–C5):")
        for name, fmap in body_outputs.items():
            print(f"  Layer '{name}' : {list(fmap.shape)}, file: fpn.py")

        # --- Step 2: FPN (top-down + lateral) ---
        fpn_outputs = backbone.fpn(body_outputs)   # OrderedDict

        print("\nFPN outputs (P2–P6):")
        for name, fmap in fpn_outputs.items():
            print(f"  FPN level '{name}' : {list(fmap.shape)}, file: fpn.py")

        # Sanity check: all FPN levels should have 256 channels
        for name, fmap in fpn_outputs.items():
            assert fmap.shape[1] == 256, \
                f"Expected 256 channels at level '{name}', got {fmap.shape[1]}"
        print("\n✓ All FPN levels have 256 channels.")

    print("="*60 + "\n")
    return body_outputs, fpn_outputs