"""
models/backbone.py

Builds a ResNet-50 + FPN backbone.
Supports three weight loading strategies controlled by ModelConfig:
  "imagenet" → ResNet-50 ImageNet weights (backbone only)
  "coco"     → full Faster R-CNN COCO weights applied to backbone
  "local"    → weights from a local .pth file
  "none"     → random initialisation

The debug_backbone() function runs a forward pass with dummy data
and prints every intermediate tensor shape so you can verify the
backbone output channels and spatial resolutions before plugging
it into the full detector.
"""
##### ------ IMPORTANT -------######## 
#In torchvision Faster R-CNN, “backbone” means the entire feature extractor stack: ResNet body + FPN together.

#################################################################################################################
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from objdet.entity.config_entity import ModelConfig


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_backbone(model_cfg: ModelConfig):
    """
    Return a BackboneWithFPN (ResNet-50 + FPN) configured per model_cfg.

    BackboneWithFPN.out_channels == 256 always (FPN output).
    It exposes 4 feature-map levels: '0', '1', '2', '3'
    (plus a 5th 'pool' level for ROI pooling, added by torchvision).

    Weight strategy
    ---------------
    "imagenet"
        resnet_fpn_backbone loads ResNet-50 with ImageNet weights.
        FPN weights are randomly initialised (standard fine-tune setup).

    "coco"
        Build the full Faster R-CNN COCO model, then extract its backbone.
        This gives FPN weights trained end-to-end on COCO, not just ImageNet.

    "local" + load_backbone_only=True
        Load a .pth produced by save_checkpoint() that contains backbone
        state under the key "backbone_state_dict" (our custom key).

    "local" + load_backbone_only=False
        Load a full Faster R-CNN state dict from disk, strip the "backbone."
        prefix, and load only the backbone sub-module weights.

    "none"
        Fully random weights. Useful for ablation studies.
    """

    strategy = model_cfg.backbone_weights.lower()
    print(f"[DEBUG] Backbone strategy: '{strategy}'")  # DEBUG: print the strategy

    if strategy == "imagenet":
        # Standard approach: ResNet-50 pretrained on ImageNet, FPN random init
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.IMAGENET1K_V1,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print("[Backbone] Loaded ResNet-50 ImageNet weights (FPN randomly initialised).")

    elif strategy == "coco":
        # Load the full COCO Faster R-CNN, then detach its backbone submodule.
        # This is the cleanest way to get FPN weights trained on COCO.
        full_model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=model_cfg.trainable_backbone_layers,
        )
        backbone = full_model.backbone
        # Apply trainable-layer freeze manually because we pulled backbone
        # out of the full model after the factory already set it.
        _apply_trainable_layers(backbone, model_cfg.trainable_backbone_layers)
        print("[Backbone] Extracted backbone from full COCO Faster R-CNN model.")

        """
        The second _apply_trainable_layers() exists because once you extract the backbone from the full Faster R-CNN model, 
        you must re-apply freezing/unfreezing logic manually, since the backbone is now independent and 
        may not fully retain the original training-layer configuration context.
        """

    elif strategy == "local":
        # Start with a randomly-initialised backbone, then load weights.
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,  # no pretrained weights yet or we will load from local .pth
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        _load_local_backbone_weights(backbone, model_cfg)

    elif strategy == "none":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print("[Backbone] Using randomly initialised backbone (no pretrained weights).")

    else:
        raise ValueError(
            f"[Backbone] Unknown backbone_weights strategy: '{strategy}'. "
            "Choose from: 'imagenet' | 'coco' | 'local' | 'none'."
        )

    return backbone


# ---------------------------------------------------------------------------
# Local weight loading helper
# ---------------------------------------------------------------------------

def _load_local_backbone_weights(backbone, model_cfg: ModelConfig):
    """
    Load backbone weights from a local .pth file.

    Two sub-cases:
    A) load_backbone_only=True
       The .pth was saved with our custom key "backbone_state_dict".
       Directly load into the backbone module.

    B) load_backbone_only=False
       The .pth is a full model checkpoint (e.g. saved by save_checkpoint).
       The full state dict has keys like "model_state_dict.backbone.body.layer1..."
       We strip the "backbone." prefix and load into the backbone.
    """
    path = model_cfg.local_weights_path
    if path is None:
        raise ValueError(
            "[Backbone] backbone_weights='local' but local_weights_path is null in config."
        )

    state = torch.load(path, map_location="cpu") #state can be either a full checkpoint dict or a backbone-only state dict

    if model_cfg.load_backbone_only:
        # Case A: state is a full checkpoint dict but we want only backbone. checkpoint saved as {"backbone_state_dict": {...}}
        if "backbone_state_dict" in state:
            backbone.load_state_dict(state["backbone_state_dict"], strict=True) #strict=True because we expect an exact match when loading backbone-only
            print(f"[Backbone] Loaded backbone-only weights from: {path}")
        else:
            # Fallback: assume the file IS the backbone state dict directly
            backbone.load_state_dict(state, strict=True)
            print(f"[Backbone] Loaded backbone state dict directly from: {path}")

    else:
        # Case B: full model checkpoint → extract "backbone.*" keys
        full_state = state.get("model_state_dict", state)

        # Keys in a full Faster R-CNN model look like:
        #   "backbone.body.layer1.0.conv1.weight"
        #   "backbone.fpn.inner_blocks.0.weight"
        # We strip the leading "backbone." to match backbone.state_dict() keys.
        backbone_state = {
            k.replace("backbone.", "", 1): v #1 means only replace the first occurrence, which is the prefix we want to remove
            for k, v in full_state.items()
            if k.startswith("backbone.")
        }

        if not backbone_state:
            raise RuntimeError(
                f"[Backbone] No 'backbone.*' keys found in checkpoint: {path}. "
                "If this is a backbone-only file, set load_backbone_only: true."
            )

        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False) #strict=False because we allow missing keys (e.g. if the checkpoint has extra keys not used by the backbone) and unexpected keys (e.g. if the checkpoint is a full model state dict with non-backbone keys)
        #load_state_dict returns two lists: missing keys (expected by backbone but not found in checkpoint) and unexpected keys (found in checkpoint but not expected by backbone). We print these for debugging.
        print(f"[Backbone] Loaded backbone weights from full checkpoint: {path}")
        if missing:
            print(f"  [Backbone] Missing keys  ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"  [Backbone] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


# ---------------------------------------------------------------------------
# Trainable-layer control helper (used after pulling backbone from full model)
# ---------------------------------------------------------------------------

def _apply_trainable_layers(backbone, trainable_layers: int):
    """
    Freeze ResNet stages so that only the last *trainable_layers* stages
    have requires_grad=True.

    ResNet-50 stages:  stem(0), layer1(1), layer2(2), layer3(3), layer4(4)
    trainable_layers=3 → freeze stem+layer1+layer2, train layer3+layer4+FPN
    This mirrors what resnet_fpn_backbone() does internally.
    """
    # Layers from stem to end — matches torchvision ordering
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "layer0"]
    layers_to_train = layers_to_train[:trainable_layers]

    for name, param in backbone.named_parameters():
        # If the parameter belongs to a layer we want to train, leave it.
        # Otherwise freeze it.
        should_train = any(name.startswith(layer) for layer in layers_to_train)
        param.requires_grad_(should_train)


# ---------------------------------------------------------------------------
# DEBUG HOOK — run this standalone to inspect backbone tensor flow
# ---------------------------------------------------------------------------

def debug_backbone(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    Run a forward pass through the backbone with dummy input.
    Prints tensor shapes at each FPN output level.

    No training, no dataset, no GPU required.

    Expected output (ResNet-50 + FPN, input 600×800):
    ──────────────────────────────────────────────────
    Input images      : [2, 3, 600, 800]
    FPN level '0'     : [2, 256, 150, 200]   ← stride 4  (layer1 output)
    FPN level '1'     : [2, 256,  75, 100]   ← stride 8  (layer2 output)
    FPN level '2'     : [2, 256,  38,  50]   ← stride 16 (layer3 output)
    FPN level '3'     : [2, 256,  19,  25]   ← stride 32 (layer4 output)
    FPN level 'pool'  : [2, 256,  10,  13]   ← stride 64 (max-pool of level 3)
    ──────────────────────────────────────────────────
    The 'pool' level is used by ROI Align for large objects.
    All levels have out_channels=256 (FPN lateral convolutions).
    """
    print("\n" + "="*60)
    print("DEBUG: Backbone tensor flow, file: backbone.py")
    print("="*60)

    # Use "none" strategy so we don't need to download weights for debug
    from objdet.entity.config_entity import ModelConfig
    cfg = ModelConfig(backbone_weights="none", trainable_backbone_layers=3)
    backbone = build_backbone(cfg)
    backbone.eval()

    # Faster R-CNN's GeneralizedTransform expects values in [0,1]
    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]

    print(f"\nInput images      : {[list(img.shape) for img in dummy_images]}, file: backbone.py")

    with torch.no_grad():
        # backbone() takes a batched tensor [B, C, H, W]
        # (the full model handles image list → tensor internally via Transform)
        dummy_batch = torch.stack(dummy_images)  # [B, 3, H, W]
        feature_maps = backbone(dummy_batch)     # OrderedDict

    print("\nFPN output feature maps:")
    for level_name, fmap in feature_maps.items():
        print(f"  FPN level '{level_name}' : {list(fmap.shape)}, file: backbone.py")

    print(f"\nBackbone out_channels : {backbone.out_channels}, file: backbone.py")
    print("="*60 + "\n")

    return feature_maps