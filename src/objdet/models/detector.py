"""
models/detector.py

Assembles complete Faster R-CNN from explicit torchvision components.

Four weight strategies, two code paths:

PATH 1 — _build_from_components()
  Used for: "imagenet", "none", "local"+load_backbone_only=True
  Steps:
    a. backbone.py builds BackboneWithFPN with weights already loaded
    b. All other components built here with random weights
    c. FasterRCNN assembled from these components

PATH 2 — _build_from_full_weights()
  Used for: "coco", "local"+load_backbone_only=False
  Steps:
    a. Build full FasterRCNN architecture with random weights
    b. Load full weight file into entire model (backbone+RPN+ROI heads)
    c. Replace box predictor head for num_classes

Tensor flow through the assembled model (eval mode):
  Input: list of [3, H, W] float tensors (one per image, values in [0,1])
      ↓
  GeneralizedRCNNTransform
      normalize: (pixel - mean) / std per channel
      resize:    shortest side → [min_size, max_size], aspect ratio kept
      batch:     zero-pad to same H'×W', return ImageList
      Output: ImageList(.tensors=[B, 3, H', W'], .image_sizes=[(H,W),...])
      ↓
  BackboneWithFPN
      ResNet body:
        C2: [B, 256,  H'/4,  W'/4]
        C3: [B, 512,  H'/8,  W'/8]
        C4: [B, 1024, H'/16, W'/16]
        C5: [B, 2048, H'/32, W'/32]
      FPN top-down + lateral:
        P2: [B, 256, H'/4,  W'/4]   stride 4
        P3: [B, 256, H'/8,  W'/8]   stride 8
        P4: [B, 256, H'/16, W'/16]  stride 16
        P5: [B, 256, H'/32, W'/32]  stride 32
        P6: [B, 256, H'/64, W'/64]  stride 64  (max-pool of P5)
      Output: OrderedDict{"0":P2, "1":P3, "2":P4, "3":P5, "pool":P6}
      ↓
  RegionProposalNetwork
      AnchorGenerator:
        generates anchors at every spatial location on every FPN level
        sizes (32,64,128,256,512) × aspect_ratios (0.5,1.0,2.0)
        total anchors ≈ 200,000 per image
      RPNHead (shared across all FPN levels):
        input: each FPN level [B, 256, H_l, W_l]
        objectness: [B, 3, H_l, W_l]  (1 score per anchor)
        bbox_delta: [B, 12, H_l, W_l] (4 values per anchor)
      Decode deltas → proposal boxes
      NMS: pre_nms=2000 → post_nms=2000 (train), 1000/1000 (test)
      Output: list of [N_proposals, 4] per image  (xyxy format)
      ↓
  RoIHeads
      MultiScaleRoIAlign:
        assigns each proposal to FPN level by box size
        bilinear interpolation → fixed 7×7 grid per proposal
        output: [total_proposals_in_batch, 256, 7, 7]
      TwoMLPHead:
        flatten: [total_proposals, 256×7×7=12544]
        FC(12544→1024) + ReLU
        FC(1024→1024)  + ReLU
        output: [total_proposals, 1024]
      FastRCNNPredictor:
        cls_score: FC(1024→num_classes) → [total_proposals, num_classes]
        bbox_pred: FC(1024→num_classes×4) → [total_proposals, num_classes×4]
      decode + per-class NMS → final detections
      Output: list of {"boxes":[K,4], "labels":[K], "scores":[K]} per image
"""

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from objdet.entity.config_entity import ModelConfig
from objdet.models.backbone import build_backbone


# ===========================================================================
# PUBLIC API
# ===========================================================================

def build_faster_rcnn(model_cfg: ModelConfig) -> FasterRCNN:
    """
    Entry point. Dispatches to the correct build path based on strategy.
    """
    strategy = model_cfg.backbone_weights.lower()
    only_backbone = model_cfg.load_backbone_only

    # PATH 1: backbone.py handles weight loading, components built here
    if strategy in ("imagenet", "none") or (strategy == "local" and only_backbone):
        return _build_from_components(model_cfg)

    # PATH 2: full model weights loaded here, backbone.py not involved
    if strategy == "coco" or (strategy == "local" and not only_backbone):
        return _build_from_full_weights(model_cfg)

    raise ValueError(
        f"Unknown backbone_weights='{strategy}'. "
        "Choose: 'imagenet' | 'coco' | 'local' | 'none'"
    )


def get_model_on_device(model_cfg: ModelConfig, device: torch.device) -> FasterRCNN:
    """Build model and move to device. Called by trainer and inference."""
    model = build_faster_rcnn(model_cfg)
    model.to(device)
    return model


# ===========================================================================
# PATH 1 — build from explicit components
# Strategies: "imagenet", "none", "local"+load_backbone_only=True
# Backbone weights already loaded by backbone.py.
# RPN + ROI heads always randomly initialised here.
# ===========================================================================

def _build_from_components(model_cfg: ModelConfig) -> FasterRCNN:

    # ── 1. Backbone ───────────────────────────────────────────────────────
    # backbone.py builds BackboneWithFPN and loads weights per strategy.
    # After this call:
    #   .body weights: ImageNet / local / random depending on strategy
    #   .fpn weights: always random
#                 (FPN layers are newly constructed, not pretrained)
    #   .out_channels: always 256
    backbone = build_backbone(model_cfg)

    # ── 2. AnchorGenerator ────────────────────────────────────────────────
    # Generates anchor boxes at every spatial location on every FPN level.
    #
    # sizes: one tuple per FPN level (must match number of FPN outputs).
    #   P2 → (32,)  : anchors of base size 32px (small objects: cyclists)
    #   P3 → (64,)  : anchors of base size 64px
    #   P4 → (128,) : anchors of base size 128px
    #   P5 → (256,) : anchors of base size 256px
    #   P6 → (512,) : anchors of base size 512px (large objects: buses, trucks)
    #
    # aspect_ratios: (0.5, 1.0, 2.0) applied to each size.
    #   0.5  → tall anchor  (h = w*2)  good for: persons, riders
    #   1.0  → square anchor           good for: cars
    #   2.0  → wide anchor  (w = h*2)  good for: buses, trucks
    #
    # Total anchors per FPN level at spatial location (x,y): 3
    # (1 size × 3 aspect ratios)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # ── 3. RPNHead ─────────────────────────────────────────────────────────
    # Shared across all FPN levels (same weights applied to each level).
    # This sharing works because FPN normalises all levels to 256 channels.
    #
    # Internal layers:
    #   conv:       Conv2d(256, 256, kernel=3, padding=1) + ReLU
    #               → [B, 256, H_l, W_l]  (same spatial size, 256ch)
    #   cls_logits: Conv2d(256, 3, kernel=1)
    #               → [B, 3, H_l, W_l]    (3 = num_anchors per location)
    #   bbox_pred:  Conv2d(256, 12, kernel=1)
    #               → [B, 12, H_l, W_l]   (12 = 3 anchors × 4 deltas)
    #
    # num_anchors_per_location()[0] = 3
    # (asking anchor_generator how many anchors it produces per location)
    rpn_head = RPNHead(
        in_channels=backbone.out_channels,                            # 256
        num_anchors=anchor_generator.num_anchors_per_location()[0],  # 3
    )

    # ── 4. RegionProposalNetwork ───────────────────────────────────────────
    # Runs anchor scoring → box decoding → NMS → proposal selection.
    #
    # Matcher thresholds (assigns GT boxes to anchors for RPN training):
    #   anchor IoU ≥ 0.7 with any GT → positive anchor (should detect object)
    #   anchor IoU < 0.3 with all GT → negative anchor (background)
    #   0.3 ≤ IoU < 0.7             → ignored during training
    #
    # Sampler (selects which anchors contribute to RPN loss):
    #   256 anchors per image
    #   up to 50% positive (128 pos + 128 neg ideally)
    #   if fewer positives, fills remainder with negatives
    #
    # NMS:
    #   pre_nms_top_n:  score-ranked anchors kept before NMS
    #   post_nms_top_n: top proposals kept after NMS
    #   nms_thresh=0.7: suppress anchors with IoU > 0.7 with a higher-scoring anchor
    #
    # score_thresh=0.0: keep all proposals regardless of score (filtering
    # happens later in ROI heads by score_thresh=0.05)
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 2000, "testing": 1000},
        post_nms_top_n={"training": 2000, "testing": 1000},
        nms_thresh=0.7,
        score_thresh=0.0,
    )

    # ── 5. MultiScaleRoIAlign ──────────────────────────────────────────────
    # Extracts fixed-size feature grids for each proposal from FPN.
    #
    # Level assignment (which FPN level to pool from for each proposal):
    #   k = clip( floor(4 + log2(sqrt(w×h) / 224)), min=2, max=5 )
    #   small proposal  (e.g. 32×32)  → k=2 → pool from P2 (high resolution)
    #   medium proposal (e.g. 128×128)→ k=3 → pool from P3
    #   large proposal  (e.g. 512×512)→ k=5 → pool from P5 (semantic features)
    #
    # featmap_names=["0","1","2","3"]: FPN levels to use (P2..P5)
    #   "pool" (P6) excluded: P6 is only for RPN anchor generation (large anchors)
    #   not for ROI pooling (proposals at that scale are too large for 7×7 pool)
    #
    # output_size=7: each proposal is extracted as a 7×7 feature grid
    #
    # sampling_ratio=2: 2×2=4 bilinear sample points per output cell
    #   improves alignment accuracy vs nearest-neighbor
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ── 6. TwoMLPHead ──────────────────────────────────────────────────────
    # Processes each ROI-pooled feature into a 1024-dim vector.
    #
    # Input shape:  [N_proposals, 256, 7, 7]
    # After flatten: [N_proposals, 256×7×7] = [N_proposals, 12544]
    # FC1:           [N_proposals, 12544] → [N_proposals, 1024] + ReLU
    # FC2:           [N_proposals, 1024]  → [N_proposals, 1024] + ReLU
    # Output:        [N_proposals, 1024]
    #
    # in_channels = backbone.out_channels × 7 × 7 = 256 × 49 = 12544
    box_head = TwoMLPHead(
        in_channels=backbone.out_channels * 7 * 7,  # 256 × 7 × 7 = 12544
        representation_size=1024,
    )

    # ── 7. FastRCNNPredictor ───────────────────────────────────────────────
    # Two parallel FC heads producing final class scores and box offsets.
    #
    # Input:     [N_proposals, 1024]
    # cls_score: FC(1024, num_classes)     → [N_proposals, num_classes]
    #            one score per class (including background=class 0)
    # bbox_pred: FC(1024, num_classes × 4) → [N_proposals, num_classes × 4]
    #            separate box delta for each class
    #            at inference: use predicted class index to select the right 4
    box_predictor = FastRCNNPredictor(
        in_channels=1024,
        num_classes=model_cfg.num_classes,  # 9 for Cityscapes (8 + background)
    )

    # ── 8. RoIHeads ────────────────────────────────────────────────────────
    # Orchestrates proposal assignment, sampling, head forward, loss/decode.
    #
    # Matcher (assigns GT boxes to proposals for ROI training):
    #   proposal IoU ≥ fg_iou_thresh=0.5 → positive (train to detect object)
    #   proposal IoU < bg_iou_thresh=0.5 → negative (train as background)
    #   Note: fg=bg=0.5 means no ignored region (every proposal is pos or neg)
    #
    # Sampler:
    #   512 proposals per image
    #   positive_fraction=0.25 → up to 128 positive + 384 negative per image
    #   If fewer than 128 positives, fills remainder with negatives
    #
    # bbox_reg_weights=None → uses default (10.0, 10.0, 5.0, 5.0)
    #   These scale the regression targets during training to normalize magnitude.
    #   Higher weights for x,y (10) vs w,h (5): center regression more important.
    #
    # Post-processing (inference only, no effect during training):
    #   score_thresh=0.05: drop detections with confidence < 5%
    #   nms_thresh=0.5:    per-class NMS, suppress IoU > 0.5
    #   detections_per_img=100: max final detections returned per image
    roi_heads = RoIHeads(
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )

    # ── 9. GeneralizedRCNNTransform ───────────────────────────────────────
    # Applied BEFORE backbone. Three sequential operations:
    #
    # 1. Normalize per channel: pixel = (pixel - mean) / std
    #    Using ImageNet statistics even for random-init models is standard.
    #    Keeps pixel values in a range that works well with ResNet BatchNorm.
    #
    # 2. Resize: scale so shortest side is in [min_size, max_size] range.
    #    aspect ratio preserved. Longer side capped at max_size.
    #    Cityscapes images are 2048×1024 → resized to ~800×1600 (capped at 1333)
    #
    # 3. Batch: zero-pad all images in batch to the same H'×W'.
    #    Returns ImageList with .tensors=[B,3,H',W'] and .image_sizes=list
    transform = GeneralizedRCNNTransform(
        min_size=model_cfg.min_size,       # 800
        max_size=model_cfg.max_size,       # 1333
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # ── 10. FasterRCNN (final assembly) ───────────────────────────────────
    # Wraps all components. forward() orchestrates the full pipeline:
    #   train mode: (images, targets) → loss_dict
    #     {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    #   eval mode:  (images,) → list of dicts per image
    #     [{"boxes":[K,4], "labels":[K], "scores":[K]}, ...]
    #
    # num_classes=None: we pass box_predictor directly, so FasterRCNN should
    # NOT create its own predictor. Passing num_classes would override ours.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None,
        rpn=rpn,
        roi_heads=roi_heads,
        transform=transform,
    )

    print(
        f"[Detector] Built from components | "
        f"strategy='{model_cfg.backbone_weights}' | "
        f"num_classes={model_cfg.num_classes}"
    )
    return model


# ===========================================================================
# PATH 2 — load full model weights
# Strategies: "coco", "local"+load_backbone_only=False
# Builds architecture first (random weights), then loads full checkpoint.
# ===========================================================================

def _build_from_full_weights(model_cfg: ModelConfig) -> FasterRCNN:
    """
    Build the full FasterRCNN architecture, load a complete weight file
    into it (backbone + RPN + ROI heads), then replace the box predictor
    for our num_classes.

    Why build architecture first, then load?
    PyTorch's load_state_dict() requires the model to already exist.
    You cannot load weights into nothing — the nn.Module structure
    must be constructed first, then weights are copied into it.
    """
    strategy = model_cfg.backbone_weights.lower()

    # ── Build architecture (random weights) ────────────────────────────────
    # We use the torchvision factory here (weights=None) because for "coco"
    # we need the architecture to exactly match the COCO checkpoint's key names.
    # The factory guarantees this. For "local"+load_backbone_only=False we need the same
    # architecture as whatever produced the local .pth.
    model = fasterrcnn_resnet50_fpn(
        weights=None,                  # random weights, will be overwritten below
        min_size=model_cfg.min_size,
        max_size=model_cfg.max_size,
        trainable_backbone_layers=model_cfg.trainable_backbone_layers,
    )

    # ── Load full weights ──────────────────────────────────────────────────
    if strategy == "coco":
        # Download and load official COCO pretrained weights.
        # This sets weights for: backbone (body+FPN) + RPN + ROI heads
        # # Full COCO weights are loaded first, including the original
        # COCO box predictor (91 classes). We replace the predictor next.
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model.load_state_dict(weights.get_state_dict(progress=True))
        print(
            "[Detector] strategy='coco' | "
            "Full COCO pretrained weights loaded (backbone+RPN+ROI heads)"
        )

    elif strategy == "local" and not model_cfg.load_backbone_only:
        # Load full model weights from local file.
        # File must have been saved by save_checkpoint() (our format)
        # or be a raw torchvision state dict.
        path = model_cfg.local_weights_path
        if not path:
            raise ValueError(
                "[Detector] local_weights_path is null. "
                "Set it in config for strategy='local'."
            )
        checkpoint = torch.load(path, map_location="cpu")

        # Our save_checkpoint() format wraps state dict under "model_state_dict"
        sd = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(
            f"[Detector] strategy='local'+load_backbone_only=False | "
            f"Loaded from: {path} | "
            f"missing={len(missing)} | unexpected={len(unexpected)}"
        )

    # ── Replace box predictor ──────────────────────────────────────────────
    # The loaded weights have 91 classes (COCO) or whatever the local file had.
    # We replace the final classification + regression heads for our num_classes.
    # Only these two FC layers are randomly reinitialised — everything else
    # keeps the pretrained weights.
    #
    # in_features = 1024 (output size of TwoMLPHead, same regardless of dataset)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,       # 1024
        num_classes=model_cfg.num_classes,
    )
    print(
        f"[Detector] Box predictor replaced | "
        f"in_features={in_features} | "
        f"num_classes={model_cfg.num_classes} (was 91)"
    )

    return model