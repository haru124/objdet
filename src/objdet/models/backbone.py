"""
models/backbone.py

Responsibility:
    Build BackboneWithFPN and load ONLY ResNet-body weights. 

Called by detector.py ONLY for these strategies:
  "imagenet"              → ResNet-50 ImageNet weights, FPN random
  "local" + only_backbone → local .pth backbone weights, FPN random
  "none"                  → fully random backbone + FPN

NOT called for:
  "coco"                  → detector.py loads full COCO model directly
  "local" + full_model    → detector.py loads full local .pth directly

BackboneWithFPN internal structure (torchvision):
  .body                   → IntermediateLayerGetter (ResNet-50)
      stem (conv1+bn1+relu+maxpool)
      .layer1             → 3  Bottleneck blocks → C2 [B, 256,  H/4,  W/4]
      .layer2             → 4  Bottleneck blocks → C3 [B, 512,  H/8,  W/8]
      .layer3             → 6  Bottleneck blocks → C4 [B, 1024, H/16, W/16]
      .layer4             → 3  Bottleneck blocks → C5 [B, 2048, H/32, W/32]
  .fpn                    → FeaturePyramidNetwork
      inner_blocks        → 4× Conv2d(C_in, 256, 1×1) lateral convs
      layer_blocks        → 4× Conv2d(256,  256, 3×3) output convs
      extra_blocks        → MaxPool2d on P5 → P6
      Output: P2[/4] P3[/8] P4[/16] P5[/32] P6[/64], all 256ch
  .out_channels = 256     → constant regardless of input
"""

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights

from objdet.entity.config_entity import ModelConfig


def build_backbone(model_cfg: ModelConfig):
    """
    Builds and returns BackboneWithFPN.

    Only call this for: "imagenet", "none", "local"+load_backbone_only=True
    For "coco" and "local"+load_backbone_only=False, detector.py handles
    everything and does not call this function.
    """
    strategy = model_cfg.backbone_weights.lower()

    # ── "imagenet" ─────────────────────────────────────────────────────────
    # ResNet-50 body loaded with ImageNet weights.
    # FPN lateral + output convs are randomly initialised.
    # trainable_layers controls how many ResNet stages are unfrozen:
    #   0 → freeze entire backbone (only train RPN + ROI heads)
    #   3 → freeze stem+layer1+layer2, train layer3+layer4+FPN (default)
    #   5 → train everything including stem
    if strategy == "imagenet":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.IMAGENET1K_V1,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print(
            f"[Backbone] imagenet | "
            f"ResNet-50 body: ImageNet weights | "
            f"FPN: random init | "
            f"trainable_layers={model_cfg.trainable_backbone_layers}"
        )
        return backbone

    # ── "none" ─────────────────────────────────────────────────────────────
    # Entire backbone (ResNet body + FPN) randomly initialised.
    # Useful for ablation studies to measure impact of pretraining.
    if strategy == "none":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print(
            f"[Backbone] none | "
            f"ResNet-50 body: random init | "
            f"FPN: random init"
        )
        return backbone

    # ── "local" + load_backbone_only=True ─────────────────────────────────
    # Build backbone with random weights first, then overwrite ONLY
    # the ResNet body weights from the local file.  
     # FPN weights are NEVER loaded and always remain randomly initialised.  
      # If the local file is a full model checkpoint, only 
      # "backbone.body.*" keys are extracted.  
      # RPN, ROI head, and FPN weights in the file are ignored.


    if strategy == "local" and model_cfg.load_backbone_only:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        _load_local_into_backbone(backbone, model_cfg)
        return backbone

    # If we reach here, caller made a mistake
    raise ValueError(
        f"[Backbone] build_backbone() called with strategy='{strategy}' "
        f"and load_backbone_only={model_cfg.load_backbone_only}. "
        f"This combination should be handled in detector.py, not backbone.py."
    )




def _load_local_into_backbone(backbone, model_cfg: ModelConfig):
    """
    Load ONLY ResNet body weights from a local .pth file.

    FPN weights are NEVER loaded and always remain randomly initialised.

    Supported formats:

    Format A — backbone-only checkpoint:
        {"backbone_state_dict": OrderedDict(...)}

        Possible keys:
            "body.layer1.0.conv1.weight"
            "fpn.inner_blocks.0.0.weight"

        Only "body.*" keys are loaded.
        Any "fpn.*" keys are ignored.

    Format B — full model checkpoint:
        {"model_state_dict": OrderedDict(...), ...}

        Possible keys:
            "backbone.body.layer1.0.conv1.weight"
            "backbone.fpn.inner_blocks.0.0.weight"
            "rpn.head.cls_logits.weight"
            "roi_heads.box_head.fc6.weight"

        Only "backbone.body.*" keys are extracted.
        FPN, RPN, and ROI head weights are ignored.

    Result:
        ResNet body  → loaded from checkpoint
        FPN          → random init
        RPN + ROI heads → not touched by this function at all, handled separately in detector.py
    """
    path = model_cfg.local_weights_path

    if not path:
        raise ValueError(
            "[Backbone] local_weights_path is null. "
            "Set it in config when backbone_weights='local'."
        )

    print(f"[Backbone] local+backbone_only | Loading from: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # ──────────────────────────────────────────────────────────────────
    # Format A — backbone-only checkpoint
    # Keep ONLY body.* keys
    # Ignore fpn.* keys completely
    # ──────────────────────────────────────────────────────────────────
    if "backbone_state_dict" in checkpoint:

        full_sd = checkpoint["backbone_state_dict"]

        sd = {
            k: v
            for k, v in full_sd.items()
            if k.startswith("body.")
        }

        print(
            f"[Backbone] Detected format: backbone-only checkpoint | "
            f"Loaded {len(sd)} ResNet body keys | "
            f"FPN weights ignored"
        )

    # ──────────────────────────────────────────────────────────────────
    # Format B — full model checkpoint
    # Extract ONLY backbone.body.*
    # Remove 'backbone.' prefix before loading
    # ──────────────────────────────────────────────────────────────────
    elif "model_state_dict" in checkpoint:

        full_sd = checkpoint["model_state_dict"]

        sd = {
            k[len("backbone."):]: v
            for k, v in full_sd.items()
            if k.startswith("backbone.body.")
        }

        print(
            f"[Backbone] Detected format: full model checkpoint | "
            f"Loaded {len(sd)} ResNet body keys | "
            f"FPN/RPN/ROI weights ignored"
        )

    # ──────────────────────────────────────────────────────────────────
    # Legacy raw state dict
    # ──────────────────────────────────────────────────────────────────
    else:

        full_sd = checkpoint

        sd = {
            k: v
            for k, v in full_sd.items()
            if k.startswith("body.")
        }

        print(
            f"[Backbone] Detected format: raw state dict | "
            f"Loaded {len(sd)} ResNet body keys"
        )

    # ──────────────────────────────────────────────────────────────────

    if not sd:
        raise RuntimeError(
            f"[Backbone] No ResNet body weights found in {path}."
        )

    missing, unexpected = backbone.load_state_dict(sd, strict=False)

    print(
        f"[Backbone] ResNet body loaded | "
        f"missing={len(missing)} | "
        f"unexpected={len(unexpected)}"
    )

    if missing:
        print(
            f"  Missing: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )

    if unexpected:
        print(
            f"  Unexpected: {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )