"""
onnx/inference.py

ONNXRuntime inference wrapper for the exported Faster R-CNN model.

TODO: The output parsing below assumes the ONNX graph exposes separate
      output nodes for boxes, labels, and scores.  If the export wraps
      everything in a tuple you may need to adjust `_parse_outputs`.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class ONNXDetector:
    """
    Run inference using an exported ONNX Faster R-CNN model.

    Args:
        onnx_path:      Path to the .onnx model file.
        providers:      ONNXRuntime execution providers, e.g.
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        score_threshold: Filter detections below this confidence.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[list[str]] = None,
        score_threshold: float = 0.5,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime is not installed.  Run: pip install onnxruntime"
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.score_threshold = score_threshold
        self._input_name = self.session.get_inputs()[0].name

        print(f"[ONNXDetector] Loaded model from {onnx_path}")
        print(f"[ONNXDetector] Providers: {self.session.get_providers()}")

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL image.

        Returns:
            dict with keys:
                boxes  : np.ndarray[N, 4]  xyxy
                labels : np.ndarray[N]
                scores : np.ndarray[N]
        """
        input_tensor = self._preprocess(image)
        # ONNXRuntime expects a numpy array
        outputs = self.session.run(None, {self._input_name: input_tensor})
        return self._parse_outputs(outputs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        """Convert PIL image to CHW float32 numpy array in [0, 1]."""
        img = image.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0   # HWC
        arr = arr.transpose(2, 0, 1)                     # CHW
        # Add batch dim: [1, C, H, W]
        return arr[np.newaxis, ...]

    def _parse_outputs(self, outputs: list) -> dict:
        """
        Parse raw ONNXRuntime outputs.

        TODO: Adjust indexing if the ONNX export produces different output order.
        Expected: outputs[0]=boxes, outputs[1]=labels, outputs[2]=scores
        """
        if len(outputs) < 3:
            return {"boxes": np.zeros((0, 4)), "labels": np.array([]), "scores": np.array([])}

        boxes  = outputs[0]   # [N, 4]
        labels = outputs[1]   # [N]
        scores = outputs[2]   # [N]

        # Filter by score
        keep = scores >= self.score_threshold
        return {
            "boxes":  boxes[keep],
            "labels": labels[keep],
            "scores": scores[keep],
        }
