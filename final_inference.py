"""
final_inference.py

Benchmarks a trained Faster R-CNN checkpoint across 4 inference modes:
  1. PyTorch  — CPU
  2. PyTorch  — GPU (if available)
  3. ONNX Runtime — CPU
  4. ONNX Runtime — GPU (if available, via CUDAExecutionProvider)

Also prints:
  - Model size on disk (.pth and .onnx)
  - Parameter counts (total, trainable, non-trainable)
  - Parameter breakdown by component (backbone, FPN, RPN, ROI head)
  - Memory breakdown (weights, gradients, activations estimate)

Usage:
  python final_inference.py \\
      --ckpt  outputs/checkpoints/exp_01_sgd.../best_exp_01_...pth \\
      --onnx  outputs/model_export/exp_01.onnx \\
      --exp   config/experiments/exp_01_sgd_cross_entropy_smoothl1.yaml \\
      --image path/to/any/image.jpg \\
      --runs  50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ===========================================================================
# HELPERS — MODEL SIZE & PARAM COUNTS
# ===========================================================================

def print_model_size(ckpt_path: str | Path, onnx_path: str | Path | None):
    """Print file sizes of .pth and .onnx on disk."""
    print("\n" + "="*65)
    print("  MODEL SIZE ON DISK")
    print("="*65)

    p = Path(ckpt_path)
    if p.exists():
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  PyTorch checkpoint (.pth) : {size_mb:.2f} MB   [{p.name}]")
    else:
        print(f"  PyTorch checkpoint        : NOT FOUND at {ckpt_path}")

    if onnx_path:
        o = Path(onnx_path)
        if o.exists():
            size_mb = o.stat().st_size / (1024 ** 2)
            print(f"  ONNX model        (.onnx): {size_mb:.2f} MB   [{o.name}]")
        else:
            print(f"  ONNX model                : NOT FOUND at {onnx_path}")
    print()


def print_param_counts(model: torch.nn.Module):
    """
    Print parameter counts:
      - Total / Trainable / Frozen
      - Per component: backbone, FPN, RPN, ROI head, other

    Parameter types explained:
      Weights     = stored model parameters (what takes up disk space)
      Gradients   = computed during backward pass, same count as weights
                    but only exist in memory during training
      Activations = intermediate feature maps during forward pass
                    (not parameters — depend on batch size and image size)
    """
    print("="*65)
    print("  PARAMETER COUNTS")
    print("="*65)

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable

    print(f"  Total parameters      : {total:>12,}")
    print(f"  Trainable (grad=True) : {trainable:>12,}  ← updated during training")
    print(f"  Frozen    (grad=False): {frozen:>12,}  ← fixed (backbone layers)")

    # ── Per-component breakdown ───────────────────────────────────────
    print(f"\n  {'Component':<30} {'Params':>12}  {'Trainable':>12}")
    print(f"  {'─'*55}")

    components = {
        "backbone (ResNet50)":    "backbone.body",
        "FPN (Feature Pyramid)":  "backbone.fpn",
        "RPN (Region Proposal)":  "rpn",
        "ROI Head (Box)":         "roi_heads",
        "Transform":              "transform",
    }

    accounted = 0
    for label, attr_path in components.items():
        try:
            # Navigate nested attributes (e.g. "backbone.body")
            module = model
            for part in attr_path.split("."):
                module = getattr(module, part)
            p_total = sum(p.numel() for p in module.parameters())
            p_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            accounted += p_total
            print(f"  {label:<30} {p_total:>12,}  {p_train:>12,}")
        except AttributeError:
            print(f"  {label:<30} {'(not found)':>12}")

    other = total - accounted
    if other > 0:
        print(f"  {'Other'::<30} {other:>12,}")

    # ── Memory footprint estimate ─────────────────────────────────────
    print(f"\n  MEMORY FOOTPRINT ESTIMATE (float32, 4 bytes/param)")
    print(f"  {'─'*55}")
    weights_mb = (total * 4) / (1024 ** 2)
    grads_mb   = (trainable * 4) / (1024 ** 2)
    print(f"  Weights (parameters)  : {weights_mb:>8.1f} MB")
    print(f"  Gradients (trainable) : {grads_mb:>8.1f} MB  (only during training)")
    print(f"  Activations           : depends on image size and batch size")
    print(f"                          (not stored as parameters — computed on-the-fly)")
    print()


# ===========================================================================
# HELPERS — SINGLE IMAGE INFERENCE
# ===========================================================================

def load_image_as_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load a PIL image and convert to CHW float32 tensor in [0,1]."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC float32
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return tensor.to(device)


def load_image_as_numpy(image_path: str) -> np.ndarray:
    """Load image as [1, C, H, W] float32 numpy array for ONNX Runtime."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0    # HWC
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]     # 1CHW
    return arr


# ===========================================================================
# BENCHMARK RUNNERS
# ===========================================================================

def benchmark_pytorch(model, image_tensor, device, n_runs: int, label: str) -> dict:
    """
    Time n_runs forward passes of model on a single image.
    First run is a warmup (not counted).

    Returns dict with min/max/mean/std latency in milliseconds.
    """
    print(f"\n  Running {label} ({n_runs} runs + 1 warmup) ...")
    model.eval()

    with torch.no_grad():
        # Warmup — first forward pass is slow due to JIT compilation / GPU init
        _ = model([image_tensor])

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model([image_tensor])
            if device.type == "cuda":
                torch.cuda.synchronize()  # wait for GPU to finish
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return _summarize_times(times, label)


def benchmark_onnx(onnx_path: str, image_numpy: np.ndarray,
                   providers: list[str], n_runs: int, label: str) -> dict:
    """
    Time n_runs ONNX Runtime forward passes on a single image.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  [{label}] SKIPPED — onnxruntime not installed")
        return {}

    print(f"\n  Running {label} ({n_runs} runs + 1 warmup) ...")

    available = ort.get_available_providers()
    usable = [p for p in providers if p in available]
    if not usable:
        print(f"  [{label}] SKIPPED — none of {providers} available. "
              f"Available: {available}")
        return {}

    session = ort.InferenceSession(str(onnx_path), providers=usable)
    input_name = session.get_inputs()[0].name
    print(f"  [{label}] Providers in use: {session.get_providers()}")

    # Warmup
    _ = session.run(None, {input_name: image_numpy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: image_numpy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return _summarize_times(times, label)


def _summarize_times(times: list[float], label: str) -> dict:
    arr = np.array(times)
    result = {
        "label":   label,
        "mean_ms": float(np.mean(arr)),
        "min_ms":  float(np.min(arr)),
        "max_ms":  float(np.max(arr)),
        "std_ms":  float(np.std(arr)),
        "fps":     float(1000.0 / np.mean(arr)),
    }
    return result


def print_timing_results(results: list[dict]):
    """Print a formatted comparison table of all timing results."""
    print("\n" + "="*65)
    print("  INFERENCE TIMING RESULTS  (single image, ms)")
    print("="*65)
    print(f"  {'Mode':<35} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8} {'FPS':>8}")
    print(f"  {'─'*63}")
    for r in results:
        if not r:
            continue
        print(
            f"  {r['label']:<35} "
            f"{r['mean_ms']:>7.1f}  "
            f"{r['min_ms']:>7.1f}  "
            f"{r['max_ms']:>7.1f}  "
            f"{r['std_ms']:>7.1f}  "
            f"{r['fps']:>7.1f}"
        )

    # Speedup ratios relative to PyTorch CPU
    cpu_result = next((r for r in results if r and "PyTorch CPU" in r["label"]), None)
    if cpu_result and len([r for r in results if r]) > 1:
        print(f"\n  SPEEDUP vs PyTorch CPU")
        print(f"  {'─'*40}")
        for r in results:
            if not r or r is cpu_result:
                continue
            speedup = cpu_result["mean_ms"] / r["mean_ms"]
            print(f"  {r['label']:<35} {speedup:.2f}×")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Faster R-CNN: PyTorch vs ONNX, CPU vs GPU",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", required=True,
        help="Path to .pth checkpoint (best_*.pth recommended)",
    )
    parser.add_argument(
        "--onnx", default=None,
        help="Path to exported .onnx file. If not provided, ONNX benchmarks are skipped.",
    )
    parser.add_argument(
        "--exp", default=None,
        help="Path to experiment YAML (for model config). Uses base config if not given.",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to a single image file (.jpg/.png) to use for timing.",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of timed forward passes per mode (default: 50).",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
    )
    args = parser.parse_args()

    from objdet.config.configuration import ConfigurationManager
    from objdet.models.detector import get_model_on_device, build_faster_rcnn
    from objdet.utils.checkpoint import load_checkpoint
    from objdet.utils.common import get_device

    cfg = ConfigurationManager(
        base_config_path=args.config,
        experiment_config_path=args.exp,
    ).get_config()

    # ── Header ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL INFERENCE BENCHMARK")
    print(f"{'='*65}")
    print(f"  Experiment  : {cfg.experiment_name}")
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  ONNX model  : {args.onnx or 'not provided'}")
    print(f"  Image       : {args.image}")
    print(f"  Runs        : {args.runs}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"{'='*65}")

    # ── Model size ────────────────────────────────────────────────────
    print_model_size(args.ckpt, args.onnx)

    # ── Load model and print param counts ─────────────────────────────
    cpu_device = torch.device("cpu")
    model_cpu = build_faster_rcnn(cfg.model)
    load_checkpoint(args.ckpt, model_cpu, map_location="cpu")
    model_cpu.eval()

    print_param_counts(model_cpu)

    # ── Load image ────────────────────────────────────────────────────
    image_numpy = load_image_as_numpy(args.image)   # for ONNX
    image_cpu   = load_image_as_tensor(args.image, cpu_device)

    timing_results = []

    # ── 1. PyTorch CPU ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BENCHMARKING")
    print("="*65)

    result = benchmark_pytorch(
        model_cpu, image_cpu, cpu_device, args.runs, "PyTorch CPU"
    )
    timing_results.append(result)

    # Sample prediction to show it's working
    with torch.no_grad():
        preds = model_cpu([image_cpu])
    pred = preds[0]
    keep = pred["scores"] >= args.score_threshold
    print(f"  → {keep.sum().item()} detections above threshold {args.score_threshold}")

    # ── 2. PyTorch GPU ────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        model_gpu = build_faster_rcnn(cfg.model)
        load_checkpoint(args.ckpt, model_gpu, map_location="cuda")
        model_gpu.eval()
        image_gpu = load_image_as_tensor(args.image, gpu_device)

        result = benchmark_pytorch(
            model_gpu, image_gpu, gpu_device, args.runs, "PyTorch GPU"
        )
        timing_results.append(result)

        with torch.no_grad():
            preds = model_gpu([image_gpu])
        pred = preds[0]
        keep = pred["scores"] >= args.score_threshold
        print(f"  → {keep.sum().item()} detections above threshold {args.score_threshold}")

        # Free GPU memory before ONNX GPU test
        del model_gpu, image_gpu
        torch.cuda.empty_cache()
    else:
        print("\n  [PyTorch GPU] SKIPPED — CUDA not available")

    # ── 3. ONNX CPU ───────────────────────────────────────────────────
    if args.onnx and Path(args.onnx).exists():
        result = benchmark_onnx(
            args.onnx, image_numpy,
            providers=["CPUExecutionProvider"],
            n_runs=args.runs,
            label="ONNX Runtime CPU",
        )
        timing_results.append(result)
    else:
        print("\n  [ONNX CPU] SKIPPED — no .onnx path provided or file not found")

    # ── 4. ONNX GPU ───────────────────────────────────────────────────
    if args.onnx and Path(args.onnx).exists() and torch.cuda.is_available():
        result = benchmark_onnx(
            args.onnx, image_numpy,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            n_runs=args.runs,
            label="ONNX Runtime GPU",
        )
        timing_results.append(result)
    else:
        if not torch.cuda.is_available():
            print("\n  [ONNX GPU] SKIPPED — CUDA not available")

    # ── Results table ─────────────────────────────────────────────────
    print_timing_results(timing_results)


if __name__ == "__main__":
    main()