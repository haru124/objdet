"""
onnx/export.py

Export a trained Faster R-CNN model to ONNX format.

NOTE: Exporting detection models to ONNX is non-trivial because:
  1. The model has dynamic input sizes.
  2. The output includes variable-length lists of boxes/scores.
  3. Some torchvision ops (e.g. NMS) need opset ≥ 11.

We export in eval mode with a dummy input; the resulting ONNX graph can be
used with ONNXRuntime for deployment.

TODO: Post-processing (NMS) may need to be separated from the ONNX graph
      for some runtimes.  Consider torch.onnx.export with custom opset
      mappings for production use.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_height: int = 800,
    input_width: int = 1333,
    opset_version: int = 12,
    device: Optional[torch.device] = None,
):
    """
    Export *model* to ONNX.

    Args:
        model:        Trained Faster R-CNN model.
        output_path:  Destination .onnx file.
        input_height: Height of the dummy input image.
        input_width:  Width of the dummy input image.
        opset_version: ONNX opset (≥11 required for torchvision ops).
        device:       Where to run the dummy forward pass.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    # Faster R-CNN expects a list of tensors, not a batched tensor
    dummy_input = [torch.rand(3, input_height, input_width, device=device)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX] Exporting model to {output_path} (opset {opset_version}) ...")

    try:
        torch.onnx.export(
            model,
            (dummy_input,),           # Model receives a list of images
            str(output_path),
            opset_version=opset_version,
            input_names=["images"],
            # Output names vary; Faster R-CNN returns boxes, labels, scores per image
            # TODO: map dynamic axes correctly for variable batch sizes
            do_constant_folding=True,
            verbose=False,
        )
        print(f"[ONNX] Export successful → {output_path}")
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        print(
            "[ONNX] TIP: torchvision Faster R-CNN has conditional logic that can "
            "trip ONNX tracing.  Try torch.jit.script first, or use opset ≥ 12."
        )
        raise


def verify_onnx(onnx_path: str | Path):
    """Basic ONNX model validity check using onnx.checker."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"[ONNX] Model at {onnx_path} is valid.")
    except ImportError:
        print("[ONNX] onnx package not installed; skipping verification.")
    except Exception as e:
        print(f"[ONNX] Verification failed: {e}")
