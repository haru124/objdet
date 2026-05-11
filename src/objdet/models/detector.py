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
        print("[Detector] Replaced model.backbone with custom-loaded backbone., file: detector.py")

    # Step 3 — Replace box predictor head
    # roi_heads.box_predictor is FastRCNNPredictor(in_features=1024, num_classes=91)
    # We replace it with one sized for our dataset (e.g. 9 for Cityscapes).
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    # in_features is 1024 for ResNet-50 backbone with FPN, as the box_head outputs 1024-dim features. 
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
    Attaches hooks to EVERY major sub-module of Faster R-CNN simultaneously
    in a single forward pass, printing the complete data flow in order.

    This is the authoritative end-to-end debug. The per-module debug functions
    in backbone.py / fpn.py / rpn.py / roi_heads.py each do the same thing
    in isolation — this one does all stages in one unified pass so you see
    the complete tensor flow without switching files.

    Hooks attached (in forward-pass order):
      1. GeneralizedRCNNTransform     — image resize/normalize output
      2. backbone.body                — ResNet body outputs C2–C5
      3. backbone.fpn inner_blocks    — lateral 1×1 conv per level
      4. backbone.fpn layer_blocks    — output 3×3 conv per level
      5. backbone.fpn                 — full FPN output P2–P6
      6. rpn.anchor_generator         — generated anchors
      7. rpn.head                     — objectness logits + bbox deltas
      8. rpn                          — proposals after NMS
      9. roi_heads.box_roi_pool       — ROI Align output
     10. roi_heads.box_head           — TwoMLPHead output
     11. roi_heads.box_predictor      — class logits + box deltas
     12. roi_heads                    — final detections
    """
    print("\n" + "#" * 65)
    print("# FULL DETECTOR DEBUG — complete tensor flow (all hooks)")
    print("# Every shape printed is from a REAL tensor, not hardcoded.")
    print("#" * 65)

    cfg = ModelConfig(backbone_weights="none", num_classes=9)
    model = build_faster_rcnn(cfg)
    model.eval()

    captured = {}
    hooks = []
    # Track insertion order for ordered printing
    order = []

    def _hook(name, shape_fn):
        """Register a hook that calls shape_fn(module, inp, output) → dict."""
        def _inner(module, inp, output):
            result = shape_fn(module, inp, output)
            captured[name] = result
            order.append(name)
        return _inner

    # ── 1. Transform ────────────────────────────────────────────────────────
    def transform_shape(module, inp, output):
        # output is an ImageList; .tensors is the batched tensor
        return {
            "batched_tensor": list(output.tensors.shape),
            "image_sizes": list(output.image_sizes),
        }
    hooks.append(model.transform.register_forward_hook(
        _hook("1_transform", transform_shape)
    ))

    # ── 2. ResNet body ───────────────────────────────────────────────────────
    def body_shape(module, inp, output):
        return {name: list(fmap.shape) for name, fmap in output.items()}
    hooks.append(model.backbone.body.register_forward_hook(
        _hook("2_resnet_body", body_shape)
    ))

    # ── 3+4. FPN inner + layer blocks ───────────────────────────────────────
    for i, block in enumerate(model.backbone.fpn.inner_blocks):
        def make_inner(idx):
            def shape_fn(module, inp, output):
                return {"in": list(inp[0].shape), "out": list(output.shape)}
            return shape_fn
        hooks.append(block.register_forward_hook(
            _hook(f"3_fpn_inner_block_{i}", make_inner(i))
        ))

    for i, block in enumerate(model.backbone.fpn.layer_blocks):
        def make_layer(idx):
            def shape_fn(module, inp, output):
                return {"in": list(inp[0].shape), "out": list(output.shape)}
            return shape_fn
        hooks.append(block.register_forward_hook(
            _hook(f"4_fpn_layer_block_{i}", make_layer(i))
        ))

    # ── 5. Full FPN output ───────────────────────────────────────────────────
    def fpn_shape(module, inp, output):
        return {level: list(fmap.shape) for level, fmap in output.items()}
    hooks.append(model.backbone.fpn.register_forward_hook(
        _hook("5_fpn_output", fpn_shape)
    ))

    # ── 6. Anchor generator ──────────────────────────────────────────────────
    def anchor_shape(module, inp, output):
        return {
            f"image_{i}": list(a.shape)
            for i, a in enumerate(output)
        }
    hooks.append(model.rpn.anchor_generator.register_forward_hook(
        _hook("6_anchor_generator", anchor_shape)
    ))

    # ── 7. RPN head ──────────────────────────────────────────────────────────
    def rpn_head_shape(module, inp, output):
        logits, deltas = output
        return {
            "fpn_inputs":         [list(f.shape) for f in inp[0]],
            "objectness_logits":  [list(t.shape) for t in logits],
            "bbox_deltas":        [list(t.shape) for t in deltas],
        }
    hooks.append(model.rpn.head.register_forward_hook(
        _hook("7_rpn_head", rpn_head_shape)
    ))

    # ── 8. Full RPN (proposals after NMS) ───────────────────────────────────
    def rpn_shape(module, inp, output):
        boxes, losses = output
        return {
            f"proposals_image_{i}": list(b.shape)
            for i, b in enumerate(boxes)
        }
    hooks.append(model.rpn.register_forward_hook(
        _hook("8_rpn_proposals", rpn_shape)
    ))

    # ── 9. ROI Align ─────────────────────────────────────────────────────────
    def roi_pool_shape(module, inp, output):
        proposals = inp[1]
        return {
            "proposals_per_image": [list(p.shape) for p in proposals],
            "total_proposals": sum(p.shape[0] for p in proposals),
            "roi_align_output": list(output.shape),
        }
    hooks.append(model.roi_heads.box_roi_pool.register_forward_hook(
        _hook("9_roi_align", roi_pool_shape)
    ))

    # ── 10. TwoMLPHead ───────────────────────────────────────────────────────
    def box_head_shape(module, inp, output):
        return {"input": list(inp[0].shape), "output": list(output.shape)}
    hooks.append(model.roi_heads.box_head.register_forward_hook(
        _hook("10_two_mlp_head", box_head_shape)
    ))

    # ── 11. FastRCNNPredictor ────────────────────────────────────────────────
    def predictor_shape(module, inp, output):
        cls_logits, box_deltas = output
        return {
            "input":        list(inp[0].shape),
            "class_logits": list(cls_logits.shape),
            "box_deltas":   list(box_deltas.shape),
        }
    hooks.append(model.roi_heads.box_predictor.register_forward_hook(
        _hook("11_box_predictor", predictor_shape)
    ))

    # ── 12. Full ROI Heads output ────────────────────────────────────────────
    def roi_heads_shape(module, inp, output):
        detections, _ = output
        return {
            f"image_{i}": {
                "boxes":  list(d["boxes"].shape),
                "labels": list(d["labels"].shape),
                "scores": list(d["scores"].shape),
            }
            for i, d in enumerate(detections)
        }
    hooks.append(model.roi_heads.register_forward_hook(
        _hook("12_roi_heads_output", roi_heads_shape)
    ))

    # ── Real forward pass ────────────────────────────────────────────────────
    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]

    print(f"\nRunning forward pass: {batch_size} images "
          f"[3, {image_height}, {image_width}] ...")

    with torch.no_grad():
        predictions = model(dummy_images)

    for h in hooks:
        h.remove()

    # ── Print in forward-pass order ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Complete tensor flow (shapes from real tensors):")
    print(f"{'─'*65}\n")

    section_titles = {
        "1_transform":       "① GeneralizedRCNNTransform  (resize + normalize)",
        "2_resnet_body":     "② ResNet-50 Body  (bottom-up pathway: C2→C5)",
        "5_fpn_output":      "③ FPN Output  (top-down pathway: P2→P6)",
        "6_anchor_generator":"④ AnchorGenerator  (multi-scale anchors)",
        "7_rpn_head":        "⑤ RPN Head  (objectness + box deltas per level)",
        "8_rpn_proposals":   "⑥ RPN NMS Output  (proposals per image)",
        "9_roi_align":       "⑦ MultiScaleRoIAlign  (ROI Align per proposal)",
        "10_two_mlp_head":   "⑧ TwoMLPHead  (FC 12544→1024→1024)",
        "11_box_predictor":  "⑨ FastRCNNPredictor  (class logits + box deltas)",
        "12_roi_heads_output":"⑩ ROI Heads Final Output  (per image detections)",
    }

    # Print stages in logical order (skip internal FPN blocks for brevity,
    # they are shown in detail in debug_fpn())
    print_keys = [
        "1_transform", "2_resnet_body", "5_fpn_output",
        "6_anchor_generator", "7_rpn_head", "8_rpn_proposals",
        "9_roi_align", "10_two_mlp_head", "11_box_predictor",
        "12_roi_heads_output",
    ]

    for key in print_keys:
        if key not in captured:
            continue
        title = section_titles.get(key, key)
        print(f"  {title}")
        data = captured[key]

        if key == "1_transform":
            print(f"    Batched tensor  : {data['batched_tensor']}")
            print(f"    Image sizes     : {data['image_sizes']}")

        elif key == "2_resnet_body":
            for stage, shape in data.items():
                stage_map = {"0": "C2", "1": "C3", "2": "C4", "3": "C5"}
                label = stage_map.get(stage, stage)
                print(f"    {stage} ({label}) → {shape}")

        elif key == "5_fpn_output":
            for level, shape in data.items():
                print(f"    P{level} → {shape}  "
                      f"(stride {2**(int(level)+2) if level.isdigit() else '?'})")

        elif key == "6_anchor_generator":
            for img_key, shape in data.items():
                n_anchors = shape[0]
                print(f"    {img_key} → {shape}  ({n_anchors:,} anchors across all levels)")

        elif key == "7_rpn_head":
            print(f"    FPN level inputs:")
            for i, shape in enumerate(data["fpn_inputs"]):
                print(f"      level {i} → {shape}")
            print(f"    Objectness logits (per level):")
            for i, shape in enumerate(data["objectness_logits"]):
                n_anchors = shape[1]
                hi, wi = shape[2], shape[3]
                print(f"      level {i} → {shape}  "
                      f"({n_anchors} anchors × {hi}×{wi} = {n_anchors*hi*wi} scores)")
            print(f"    Bbox deltas (per level):")
            for i, shape in enumerate(data["bbox_deltas"]):
                print(f"      level {i} → {shape}  ({shape[1]//4} anchors × 4 coords)")

        elif key == "8_rpn_proposals":
            for img_key, shape in data.items():
                print(f"    {img_key} → {shape}")

        elif key == "9_roi_align":
            print(f"    Proposals per image : {data['proposals_per_image']}")
            print(f"    Total proposals     : {data['total_proposals']}")
            print(f"    ROI Align output    : {data['roi_align_output']}")
            s = data["roi_align_output"]
            print(f"    → [{s[0]} proposals, {s[1]} ch, {s[2]}×{s[3]} pooled] "
                  f"(flattens to {s[1]*s[2]*s[3]} per proposal)")

        elif key == "10_two_mlp_head":
            print(f"    Input  → {data['input']}")
            print(f"    Output → {data['output']}")

        elif key == "11_box_predictor":
            print(f"    Input        → {data['input']}")
            print(f"    class_logits → {data['class_logits']}  "
                  f"({data['class_logits'][1]} classes)")
            print(f"    box_deltas   → {data['box_deltas']}  "
                  f"({data['box_deltas'][1]//4} classes × 4 coords)")

        elif key == "12_roi_heads_output":
            for img_key, det in data.items():
                n = det["boxes"][0]
                print(f"    {img_key}: {n} detections | "
                      f"boxes {det['boxes']} | labels {det['labels']} | scores {det['scores']}")

        print()

    print("#" * 65)
    print("# DEBUG COMPLETE — all shapes verified from real forward pass")
    print("#" * 65 + "\n")

    return captured, predictions