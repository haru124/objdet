"""
models/roi_heads.py

This module does NOT reimplement ROI heads.
It wraps torchvision's RoIHeads with debug utilities.

ROI Heads Architecture recap
─────────────────────────────
Input:  proposals from RPN  → list of [N_i, 4] boxes per image
        FPN feature maps    → OrderedDict of [B, 256, Hi, Wi] #P2..P5

Step 1 — ROI Align (MultiScaleRoIAlign)
  For each proposal, extract a 7×7 feature from the appropriate FPN level.
  FPN level assignment: level = floor(4 + log2(sqrt(wh) / 224)) #wh = proposal box area - width × height -- [x1,y1,x2,y2] compute: w = x2-x1 , h = y2-y1
  Clamped to levels [2, 5].
  Output: [total_proposals, 256, 7, 7]
  #rpn returns before nms proposals:
        #   [B, 3, Hi, Wi]  — 3 anchors per location, 1 score each
        #   [B, 12, Hi, Wi] — 3 anchors × 4 (dx,dy,dw,dh) deltas

        
####################-------- ROI Align Working ---------------###################
Suppose proposal: [100,50,300,250] (width=200, height=200, area=40000) sqrt(40000) = 200 
assigned to: P3

ROI Align:

*Step A : Maps image coordinates to feature-map coordinates. Suppose P3 stride = 8.
Then:
[100,50,300,250]
↓ divide by 8
[12.5,6.25,37.5,31.25]
Now ROI exists on P3 feature map.

*Step B : Crop that region from: [256,H,W] feature map. Maybe cropped region becomes: [256,25,25]

*Step C : Resize/interpolate to: [256,7,7]

Repeat for ALL proposals, If total proposals = 2000, 2000 times: crop → resize
##################----------------------------------##################

Step 2 — Box Head (TwoMLPHead)
  Flatten → FC(256×7×7 → 1024) → ReLU → FC(1024 → 1024) → ReLU
  Output: [total_proposals, 1024]

Step 3 — Box Predictor (FastRCNNPredictor)
  class scores : FC(1024 → num_classes)     → [total_proposals, num_classes] #eg,[total_proposals, 9] for Cityscapes
  box deltas   : FC(1024 → num_classes × 4) → [total_proposals, num_classes*4] #eg,[total_proposals, 36] for Cityscapes (9 classes × 4 coords)

Step 4 — Post-processing (inference only)
  box_coder.decode(deltas, proposals) → final boxes
  softmax(class_scores) → class probabilities
  NMS per class → final detections -- boxes, labels, scores, typically top 100 per image. 

Tensor flow summary
───────────────────
proposals [N_i, 4]  +  FPN features {P2..P5}
         ↓
MultiScaleRoIAlign  →  [ΣN_i, 256, 7, 7]
         ↓
TwoMLPHead          →  [ΣN_i, 1024]
         ↓
FastRCNNPredictor   →  class_logits[ΣN_i, C],  box_deltas[ΣN_i, C×4]
         ↓
(inference) decode + NMS → boxes[K,4], labels[K], scores[K]
"""

import torch
from objdet.entity.config_entity import ModelConfig


def get_roi_heads_from_model(model):
    """Return the RoIHeads sub-module from a Faster R-CNN model."""
    return model.roi_heads


def debug_roi_heads(image_height: int = 600, image_width: int = 800, batch_size: int = 2):
    """
    Hooks into every sub-module of RoIHeads:
      - roi_heads.box_roi_pool   → MultiScaleRoIAlign output
      - roi_heads.box_head       → TwoMLPHead output
      - roi_heads.box_predictor  → FastRCNNPredictor output
      - roi_heads                → full RoIHeads output (final detections)

    Also captures the proposal boxes fed INTO roi_heads from RPN via
    a hook on the RPN output, so we can show the full data flow.

    Every shape comes from real tensors during the actual forward pass.
    """
    from objdet.models.detector import build_faster_rcnn

    print("\n" + "=" * 65)
    print("DEBUG: ROI Heads — real tensor shapes from forward hooks")
    print("=" * 65)

    cfg = ModelConfig(backbone_weights="none", num_classes=9)
    model = build_faster_rcnn(cfg)
    model.eval()

    captured = {}
    hooks = []

    # ------------------------------------------------------------------
    # Capture RPN output proposals before they enter ROI heads
    # RPN hook → proposals = list of [N_i, 4] per image
    # ------------------------------------------------------------------
    def rpn_out_hook(module, inp, output):
        boxes, _ = output   # output = (proposals, losses)
        captured["rpn_proposals"] = [list(b.shape) for b in boxes]
        captured["rpn_proposal_total"] = sum(b.shape[0] for b in boxes)

    hooks.append(model.rpn.register_forward_hook(rpn_out_hook))

    # ------------------------------------------------------------------
    # Hook on MultiScaleRoIAlign (box_roi_pool)
    # Input:  feature_maps (dict) + proposals (list of boxes) + image_sizes
    # Output: [total_proposals_in_batch, C, pool_h, pool_w]
    #         where pool_h = pool_w = 7 (default for Faster R-CNN)
    # ------------------------------------------------------------------
    def roi_pool_hook(module, inp, output):
        # inp[0] = feature_maps dict, inp[1] = proposal boxes, inp[2] = image_sizes
        # We capture the input proposal boxes to show what goes IN
        proposals_in = inp[1]   # list of [N_i, 4] per image
        captured["roi_pool_proposals_in"] = [list(p.shape) for p in proposals_in]
        captured["roi_pool_total_proposals"] = sum(p.shape[0] for p in proposals_in)
        captured["roi_pool_output"] = list(output.shape)
        # output: [total_proposals, 256, 7, 7]

    hooks.append(model.roi_heads.box_roi_pool.register_forward_hook(roi_pool_hook))

    # ------------------------------------------------------------------
    # Hook on TwoMLPHead (box_head)
    # Input:  [total_proposals, 256, 7, 7]  (flattened internally to 256*7*7=12544)
    # Output: [total_proposals, 1024]
    # ------------------------------------------------------------------
    def box_head_hook(module, inp, output):
        captured["box_head_input"]  = list(inp[0].shape)
        captured["box_head_output"] = list(output.shape)

    hooks.append(model.roi_heads.box_head.register_forward_hook(box_head_hook))

    # ------------------------------------------------------------------
    # Hook on FastRCNNPredictor (box_predictor)
    # Input:  [total_proposals, 1024]
    # Output: (class_logits [total, num_classes],
    #          box_deltas   [total, num_classes*4])
    # ------------------------------------------------------------------
    def box_predictor_hook(module, inp, output):
        class_logits, box_deltas = output
        captured["predictor_input"]        = list(inp[0].shape)
        captured["predictor_class_logits"] = list(class_logits.shape)
        captured["predictor_box_deltas"]   = list(box_deltas.shape)
        captured["num_classes"]            = class_logits.shape[1]
        captured["num_classes_x4"]         = box_deltas.shape[1]

    hooks.append(model.roi_heads.box_predictor.register_forward_hook(box_predictor_hook))

    # ------------------------------------------------------------------
    # Hook on full RoIHeads module
    # Output in eval mode: list of dicts per image
    # Each dict: {"boxes": [K,4], "labels": [K], "scores": [K]}
    # ------------------------------------------------------------------
    def roi_heads_out_hook(module, inp, output):
        detections, _ = output   # (detections_list, losses_dict)
        captured["final_detections"] = [
            {
                "boxes":  list(d["boxes"].shape),
                "labels": list(d["labels"].shape),
                "scores": list(d["scores"].shape),
                "n_dets": d["boxes"].shape[0],
            }
            for d in detections
        ]

    hooks.append(model.roi_heads.register_forward_hook(roi_heads_out_hook))

    # ------------------------------------------------------------------
    # Run real forward pass
    # ------------------------------------------------------------------
    dummy_images = [
        torch.rand(3, image_height, image_width)
        for _ in range(batch_size)
    ]

    with torch.no_grad():
        predictions = model(dummy_images)

    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Print all captured real shapes
    # ------------------------------------------------------------------
    print(f"\nInput: {batch_size} images [{3}, {image_height}, {image_width}]")

    print(f"\nRPN → ROI Heads handoff:")
    for i, shape in enumerate(captured.get("rpn_proposals", [])):
        print(f"  Image {i} proposals from RPN : {shape}")
    print(f"  Total proposals (all images) : {captured.get('rpn_proposal_total', '?')}")

    print(f"\nMultiScaleRoIAlign (box_roi_pool):")
    for i, shape in enumerate(captured.get("roi_pool_proposals_in", [])):
        print(f"  Proposals in — image {i}    : {shape}")
    print(f"  Total proposals fed in       : {captured.get('roi_pool_total_proposals', '?')}")
    print(f"  ROI Align output             : {captured.get('roi_pool_output', '?')}")
    print(f"  → [total_proposals, C, pool_h, pool_w]")
    if "roi_pool_output" in captured:
        s = captured["roi_pool_output"]
        print(f"  → [{s[0]} proposals, {s[1]} channels, {s[2]}×{s[3]} pooled]")
        print(f"  → will be flattened to {s[1]*s[2]*s[3]} features per proposal")

    print(f"\nTwoMLPHead (box_head):")
    print(f"  Input  : {captured.get('box_head_input', '?')}")
    print(f"  Output : {captured.get('box_head_output', '?')}")
    print(f"  → FC({captured['box_head_input'][1] if 'box_head_input' in captured else '?'}"
          f"→1024) → ReLU → FC(1024→1024) → ReLU")

    print(f"\nFastRCNNPredictor (box_predictor):")
    print(f"  Input        : {captured.get('predictor_input', '?')}")
    print(f"  class_logits : {captured.get('predictor_class_logits', '?')}")
    print(f"  box_deltas   : {captured.get('predictor_box_deltas', '?')}")
    n_cls = captured.get("num_classes", "?")
    print(f"  → {n_cls} classes  |  {n_cls}×4={captured.get('num_classes_x4','?')} box coords")

    print(f"\nFinal detections per image (post decode + NMS):")
    for i, det in enumerate(captured.get("final_detections", [])):
        print(f"  Image {i}:")
        print(f"    boxes  : {det['boxes']}   ({det['n_dets']} detections)")
        print(f"    labels : {det['labels']}")
        print(f"    scores : {det['scores']}")

    print("=" * 65 + "\n")
    return captured, predictions