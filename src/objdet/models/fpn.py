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


def debug_fpn(image_height: int = 600, image_width: int = 800, batch_size: int = 2):
    """
    Hooks into backbone.body and backbone.fpn separately to show exactly
    what each sub-module receives and produces.

    Hooks attached to:
      - backbone.body          → captures C2/C3/C4/C5 as a dict
      - each FPN inner_block   → lateral conv output per level
      - each FPN layer_block   → 3x3 conv output per level
      - backbone.fpn           → final FPN output per level

    Every shape is read from actual tensors during the real forward pass.
    """
    from objdet.models.backbone import build_backbone

    print("\n" + "=" * 65)
    print("DEBUG: FPN — real tensor shapes from forward hooks")
    print("=" * 65)

    cfg = ModelConfig(backbone_weights="none")
    backbone = build_backbone(cfg)
    backbone.eval()

    captured = {}
    hooks = []

    # ------------------------------------------------------------------
    # Hook on backbone.body to capture all 4 ResNet stage outputs at once
    # backbone.body.forward returns an OrderedDict of feature maps
    # ------------------------------------------------------------------
    def body_hook(module, inp, output):
        # output is OrderedDict: {"layer1": tensor, ...} or {"0":...}
        # IntermediateLayerGetter renames layers to their return_layers keys
        for name, fmap in output.items():
            captured[f"body_out_{name}"] = fmap.shape

    hooks.append(backbone.body.register_forward_hook(body_hook))

    # ------------------------------------------------------------------
    # Hook on each FPN inner_block (lateral 1×1 convs)
    # inner_blocks is a ModuleList; one conv per FPN level
    # ------------------------------------------------------------------
    for i, inner_block in enumerate(backbone.fpn.inner_blocks):
        def make_inner_hook(idx):
            def hook(module, inp, output):
                captured[f"fpn_inner_block_{idx}_in"]  = inp[0].shape
                captured[f"fpn_inner_block_{idx}_out"] = output.shape
            return hook
        hooks.append(inner_block.register_forward_hook(make_inner_hook(i)))

    # ------------------------------------------------------------------
    # Hook on each FPN layer_block (output 3×3 convs)
    # ------------------------------------------------------------------
    for i, layer_block in enumerate(backbone.fpn.layer_blocks):
        def make_layer_hook(idx):
            def hook(module, inp, output):
                captured[f"fpn_layer_block_{idx}_in"]  = inp[0].shape
                captured[f"fpn_layer_block_{idx}_out"] = output.shape
            return hook
        hooks.append(layer_block.register_forward_hook(make_layer_hook(i)))

    # ------------------------------------------------------------------
    # Hook on full FPN module
    # ------------------------------------------------------------------
    def fpn_final_hook(module, inp, output):
        for level_name, fmap in output.items():
            captured[f"fpn_final_{level_name}"] = fmap.shape

    hooks.append(backbone.fpn.register_forward_hook(fpn_final_hook))

    # Run real forward pass
    dummy = torch.rand(batch_size, 3, image_height, image_width)
    with torch.no_grad():
        body_out, fpn_out = backbone.body(dummy), None
        # Run full backbone to trigger all hooks
        output = backbone(dummy)

    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Print (all shapes from real tensors)
    # ------------------------------------------------------------------
    print(f"\nInput: [{batch_size}, 3, {image_height}, {image_width}]")

    print("\nResNet body outputs (fed into FPN):")
    for key in sorted(k for k in captured if k.startswith("body_out_")):
        level = key.replace("body_out_", "")
        print(f"  {key:<30} → {list(captured[key])}")

    print("\nFPN inner_blocks (lateral 1×1 convolutions):")
    n_inner = sum(1 for k in captured if k.endswith("_out") and "inner_block" in k)
    for i in range(n_inner // 1):
        in_k  = f"fpn_inner_block_{i}_in"
        out_k = f"fpn_inner_block_{i}_out"
        if in_k in captured:
            print(f"  inner_block[{i}]  input  → {list(captured[in_k])}")
            print(f"  inner_block[{i}]  output → {list(captured[out_k])}")

    print("\nFPN layer_blocks (output 3×3 convolutions):")
    n_layer = sum(1 for k in captured if k.endswith("_out") and "layer_block" in k)
    for i in range(n_layer // 1):
        in_k  = f"fpn_layer_block_{i}_in"
        out_k = f"fpn_layer_block_{i}_out"
        if in_k in captured:
            print(f"  layer_block[{i}]  input  → {list(captured[in_k])}")
            print(f"  layer_block[{i}]  output → {list(captured[out_k])}")

    print("\nFPN final outputs (P2–P6, all 256 channels):")
    for key in sorted(k for k in captured if k.startswith("fpn_final_")):
        level = key.replace("fpn_final_", "")
        print(f"  level '{level}' → {list(captured[key])}")

    print("=" * 65 + "\n")
    return captured