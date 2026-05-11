"""
models/backbone.py

Builds a ResNet-50 + FPN backbone (BackboneWithFPN).
Supports four weight-loading strategies via ModelConfig.backbone_weights:
  "imagenet" → ResNet-50 ImageNet weights; FPN randomly initialised
  "coco"     → backbone extracted from full COCO Faster R-CNN (FPN weights included)
  "local"    → weights from a local .pth file (full model or backbone-only)
  "none"     → random initialisation (ablation / debug)

debug_backbone() attaches real PyTorch forward hooks to capture and print
the actual tensor shape at every internal stage. No hardcoded values.
"""

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

    BackboneWithFPN.out_channels == 256 always (FPN lateral conv output).
    Exposes OrderedDict with keys '0', '1', '2', '3' (+ 'pool' for ROI Align).

    Weight strategies
    -----------------
    "imagenet"
        resnet_fpn_backbone with ResNet50_Weights.IMAGENET1K_V1.
        FPN weights are randomly initialised — the standard fine-tune setup.

    "coco"
        Build the full COCO Faster R-CNN, extract its backbone sub-module.
        Gives FPN weights trained end-to-end on COCO (not just ImageNet body).
        _apply_trainable_layers() is called again after extraction because
        trainable-layer state is tied to the full model context in the factory.

    "local" + load_backbone_only=True
        .pth was saved by save_backbone_only_checkpoint(); contains key
        "backbone_state_dict". Loaded directly into the backbone sub-module.

    "local" + load_backbone_only=False
        .pth is a full Faster R-CNN checkpoint. Keys like
        "backbone.body.layer1.0.conv1.weight" are stripped of the "backbone."
        prefix and loaded into the backbone sub-module (strict=False to
        tolerate any minor key mismatches).

    "none"
        Fully random. Used for ablation studies and debug mode.
    """
    strategy = model_cfg.backbone_weights.lower()
    print(f"[Backbone] Strategy: '{strategy}'")

    if strategy == "imagenet":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.IMAGENET1K_V1,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print("[Backbone] Loaded ResNet-50 ImageNet weights (FPN randomly initialised).")

    elif strategy == "coco":
        # Load full COCO model then detach its backbone.
        # This is the cleanest way to get FPN weights trained on COCO.
        full_model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=model_cfg.trainable_backbone_layers,
        )
        backbone = full_model.backbone
        # Re-apply freeze/unfreeze because the backbone is now a standalone
        # module and may not fully retain the factory's layer-freeze state.
        _apply_trainable_layers(backbone, model_cfg.trainable_backbone_layers)
        print("[Backbone] Extracted backbone from full COCO Faster R-CNN model.")

    elif strategy == "local":
        # Start with random weights, then overwrite from the local file.
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        _load_local_backbone_weights(backbone, model_cfg)

    elif strategy == "none":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=model_cfg.trainable_backbone_layers,
        )
        print("[Backbone] Random initialisation (no pretrained weights).")

    else:
        raise ValueError(
            f"[Backbone] Unknown backbone_weights strategy: '{strategy}'. "
            "Choose from: 'imagenet' | 'coco' | 'local' | 'none'."
        )

    return backbone


# ---------------------------------------------------------------------------
# Local weight loading
# ---------------------------------------------------------------------------

def _load_local_backbone_weights(backbone, model_cfg: ModelConfig):
    """
    Load backbone weights from a local .pth file.

    Case A — load_backbone_only=True:
        File contains key "backbone_state_dict" (written by
        save_backbone_only_checkpoint). Loaded with strict=True.

    Case B — load_backbone_only=False:
        File is a full model checkpoint (written by save_checkpoint).
        Keys beginning with "backbone." are stripped and loaded with
        strict=False to tolerate minor key mismatches between COCO
        and Cityscapes model variants.
    """
    path = model_cfg.local_weights_path
    if path is None:
        raise ValueError(
            "[Backbone] backbone_weights='local' but local_weights_path is null in config."
        )

    state = torch.load(path, map_location="cpu")

    if model_cfg.load_backbone_only:
        # Case A: file is a backbone-only checkpoint
        if "backbone_state_dict" in state:
            backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        else:
            # Fallback: assume the file IS the state dict directly
            backbone.load_state_dict(state, strict=True)
        print(f"[Backbone] Loaded backbone-only weights from: {path}")

    else:
        # Case B: full model checkpoint — extract "backbone.*" keys
        full_state = state.get("model_state_dict", state)

        # Full Faster R-CNN keys look like:
        #   "backbone.body.layer1.0.conv1.weight"
        #   "backbone.fpn.inner_blocks.0.weight"
        # Strip the leading "backbone." (first occurrence only) to match
        # backbone.state_dict() keys.
        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in full_state.items()
            if k.startswith("backbone.")
        }

        if not backbone_state:
            raise RuntimeError(
                f"[Backbone] No 'backbone.*' keys found in: {path}. "
                "If this is a backbone-only file, set load_backbone_only: true."
            )

        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        print(f"[Backbone] Loaded backbone weights from full checkpoint: {path}")
        if missing:
            print(f"  [Backbone] Missing keys   ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"  [Backbone] Unexpected keys({len(unexpected)}): {unexpected[:5]} ...")


# ---------------------------------------------------------------------------
# Trainable-layer control
# ---------------------------------------------------------------------------

def _apply_trainable_layers(backbone, trainable_layers: int):
    """
    Freeze ResNet stages so only the last *trainable_layers* stages train.

    ResNet-50 stages: stem(0), layer1(1), layer2(2), layer3(3), layer4(4)

    trainable_layers=3 → freeze stem + layer1 + layer2
                         train  layer3 + layer4 + FPN

    This mirrors what resnet_fpn_backbone() does internally so behaviour
    is consistent whether we build the backbone directly or extract it
    from a full COCO model.
    """
    # Ordered from outermost (layer4) to innermost (stem/layer0)
    all_layers = ["layer4", "layer3", "layer2", "layer1", "layer0"]
    layers_to_train = set(all_layers[:trainable_layers])

    for name, param in backbone.named_parameters():
        should_train = any(name.startswith(layer) for layer in layers_to_train)
        param.requires_grad_(should_train)


# ---------------------------------------------------------------------------
# DEBUG — real forward hooks, no hardcoded shapes
# ---------------------------------------------------------------------------

def debug_backbone(
    image_height: int = 600,
    image_width: int = 800,
    batch_size: int = 2,
):
    """
    Attach PyTorch forward hooks to every ResNet stage and to the FPN,
    run one forward pass with dummy input, and print the real tensor
    shape captured at each hook.

    Hook attachment points:
      backbone.body.layer1  (ResNet C2 — stride 4)
      backbone.body.layer2  (ResNet C3 — stride 8)
      backbone.body.layer3  (ResNet C4 — stride 16)
      backbone.body.layer4  (ResNet C5 — stride 32)
      backbone.fpn          (FPN top-down output — all levels 256ch)
      backbone              (full module — captures batched input shape)

    Every printed value is read from a real tensor during the forward pass.
    Nothing is hardcoded.

    No GPU, no dataset, no checkpoint required.
    """
    print("\n" + "=" * 65)
    print("DEBUG: Backbone — real tensor shapes via forward hooks")
    print("  file: models/backbone.py :: debug_backbone()")
    print("=" * 65)

    # Use "none" so debug works without downloading weights
    cfg = ModelConfig(backbone_weights="none", trainable_backbone_layers=3)
    backbone = build_backbone(cfg)
    backbone.eval()

    # ── Storage for hook captures ──────────────────────────────────────
    captured: dict = {}
    hooks: list = []

    # ── Hook: each ResNet stage ────────────────────────────────────────
    # backbone.body is IntermediateLayerGetter wrapping the ResNet.
    # Direct children named layer1..layer4 are the four residual stages.
    resnet_stages = ["layer1", "layer2", "layer3", "layer4"]
    stage_labels  = {"layer1": "C2", "layer2": "C3",
                     "layer3": "C4", "layer4": "C5"}

    for stage_name in resnet_stages:
        layer = getattr(backbone.body, stage_name, None)
        if layer is None:
            continue

        def _make_stage_hook(name):
            # Closure captures name correctly per iteration
            def hook(module, inp, output):
                # inp[0]: input tensor to this stage [B, C_in, H, W]
                # output: output tensor               [B, C_out, H/2, W/2]
                captured[f"body_{name}_in"]  = tuple(inp[0].shape)
                captured[f"body_{name}_out"] = tuple(output.shape)
            return hook

        hooks.append(layer.register_forward_hook(_make_stage_hook(stage_name)))

    # ── Hook: FPN module ───────────────────────────────────────────────
    # FPN.forward returns an OrderedDict: {"0": P2, "1": P3, "2": P4, "3": P5}
    def _fpn_hook(module, inp, output):
        for level_name, fmap in output.items():
            captured[f"fpn_{level_name}"] = tuple(fmap.shape)

    hooks.append(backbone.fpn.register_forward_hook(_fpn_hook))

    # ── Hook: full backbone (to capture batched input shape) ───────────
    def _backbone_in_hook(module, inp, output):
        # inp[0] is the batched image tensor [B, 3, H, W]
        captured["backbone_input"] = tuple(inp[0].shape)

    hooks.append(backbone.register_forward_hook(_backbone_in_hook))

    # ── Forward pass ──────────────────────────────────────────────────
    dummy_batch = torch.rand(batch_size, 3, image_height, image_width)
    with torch.no_grad():
        feature_maps = backbone(dummy_batch)

    # Remove all hooks immediately after use
    for h in hooks:
        h.remove()

    # ── Print results ──────────────────────────────────────────────────
    print(f"\n  Input to backbone        : {list(captured.get('backbone_input', []))}")

    print(f"\n  ResNet body — bottom-up pathway (feature extraction):")
    for stage in resnet_stages:
        in_s  = list(captured.get(f"body_{stage}_in",  []))
        out_s = list(captured.get(f"body_{stage}_out", []))
        label = stage_labels[stage]
        print(f"    {stage} ({label})")
        print(f"      input  → {in_s}")
        print(f"      output → {out_s}")

    print(f"\n  FPN — top-down pathway (all levels: {backbone.out_channels} channels):")
    for level_key in sorted(k for k in captured if k.startswith("fpn_")):
        level = level_key.replace("fpn_", "")
        shape = list(captured[level_key])
        print(f"    P{level} (FPN level '{level}') → {shape}")

    # Verify channel dimension matches backbone.out_channels
    channel_errors = []
    for level_key in (k for k in captured if k.startswith("fpn_")):
        ch = captured[level_key][1]
        if ch != backbone.out_channels:
            channel_errors.append(f"{level_key}: expected {backbone.out_channels}, got {ch}")

    if channel_errors:
        print(f"\n  [ERROR] Channel mismatches: {channel_errors}")
    else:
        print(f"\n  ✓ All FPN levels verified: {backbone.out_channels} channels each.")

    print(f"  backbone.out_channels    : {backbone.out_channels}")
    print("=" * 65 + "\n")

    return feature_maps, captured