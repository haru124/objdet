"""
main.py — Unified entry point. Thin dispatcher only.

Modes:
  train     → full training loop (default)
  inference → test-set evaluation, visualization, plots
  debug     → tensor shape inspection, no data needed
  export    → ONNX export from checkpoint

Usage examples:
  python main.py --mode train --exp config/experiments/exp_01_smoke_test.yaml
  python main.py --mode train --exp config/experiments/exp_01_smoke_test.yaml --run-inference-after
  python main.py --mode inference \
      --exp config/experiments/exp_01_smoke_test.yaml \
      --ckpt outputs/checkpoints/exp_01_smoke_test/exp_01_epoch_0020_loss_0.8231.pth \
      --n-samples 8 --score-threshold 0.4
  python main.py --mode debug
  python main.py --mode export \
      --ckpt outputs/checkpoints/exp_01_smoke_test/exp_01_epoch_0020_loss_0.8231.pth \
      --onnx outputs/model_export/model.onnx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN — Cityscapes",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "inference", "debug", "export"],
        default="train",
        help=(
            "train     : run training loop\n"
            "inference : evaluate on val/test set + visualize + plot\n"
            "debug     : print tensor shapes at each stage (no data needed)\n"
            "export    : export checkpoint to ONNX\n"
        ),
    )

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to base YAML config")
    parser.add_argument("--exp", default=None,
                        help="Path to experiment override YAML")
    parser.add_argument("--seed", type=int, default=42)

    # ── Train-specific ───────────────────────────────────────────────────────
    parser.add_argument("--resume", default=None,
                        help="[train] Checkpoint path to resume training from")
    parser.add_argument("--run-inference-after", action="store_true",
                        help="[train] Automatically run inference after training finishes")

    # ── Inference / Export shared ────────────────────────────────────────────
    parser.add_argument(
        "--ckpt", default=None,
        help=(
            "[inference / export] Path to .pth checkpoint.\n"
            "If omitted in inference mode, auto-detects latest from experiment dir."
        ),
    )

    # ── Inference-specific ───────────────────────────────────────────────────
    parser.add_argument("--n-samples", type=int, default=5,
                        help="[inference] Number of test images to visualize (default: 5)")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="[inference] Score threshold for filtering predictions")
    parser.add_argument("--output-dir", default=None,
                        help="[inference] Directory to save visualizations + plots")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="[inference] Dataset split to evaluate on (default: val)")

    # ── Export-specific ──────────────────────────────────────────────────────
    parser.add_argument("--onnx", default=None,
                        help="[export] Output .onnx path. Defaults to outputs/model_export/{exp}.onnx")

    # ── Backbone extraction ──────────────────────────────────────────────────
    parser.add_argument("--save-backbone", default=None,
                        help="Load --ckpt and save its backbone weights separately")

    return parser.parse_args()


def main():
    args = parse_args()

    from objdet.config.configuration import ConfigurationManager
    from objdet.utils.common import set_seed

    cfg = ConfigurationManager(
        base_config_path=args.config,
        experiment_config_path=args.exp,
    ).get_config()

    set_seed(args.seed)
    _print_header(cfg, args)

    # ── Route ───────────────────────────────────────────────────────────────
    if args.mode == "debug":
        _run_debug(cfg)

    elif args.mode == "train":
        best_ckpt = _run_train(args, cfg)
        if args.run_inference_after:
            print("\n" + "=" * 65)
            print("  --run-inference-after: launching inference on best checkpoint")
            print("=" * 65)
            _run_inference(args, cfg, ckpt_override=best_ckpt)

    elif args.mode == "inference":
        _run_inference(args, cfg)

    elif args.mode == "export":
        _run_export(args, cfg)

    # Backbone extraction is a post-step, works alongside any mode
    if args.save_backbone and args.ckpt:
        _save_backbone(args, cfg)


# ===========================================================================
# PIPELINE RUNNERS
# ===========================================================================

def _run_debug(cfg):
    """
    Print tensor shapes at every stage of the forward pass.
    Uses dummy input — no dataset, no GPU, no checkpoint required.

    Reads image dimensions from cfg.debug so you can control
    the dummy input size from config.yaml / exp YAML.
    """
    from objdet.models.detector import debug_detector
    debug_detector(
        image_height=cfg.debug.image_height,
        image_width=cfg.debug.image_width,
        batch_size=cfg.debug.batch_size,
    )


def _run_train(args, cfg) -> Path:
    """
    Full training loop.
    Returns path to the latest (best) checkpoint for optional chaining.
    """
    from objdet.utils.common import get_device, count_parameters, flat_config_dict
    from objdet.datasets.dataloader import build_train_val_loaders
    from objdet.models.detector import get_model_on_device
    from objdet.losses.losses import patch_roi_head_losses
    from objdet.training.trainer import Trainer
    from objdet.tracking.tensorboard_logger import TensorBoardLogger
    from objdet.tracking.mlflow_logger import MLflowLogger
    from objdet.utils.checkpoint import get_latest_checkpoint
    from objdet.profiler.profiler_utils import build_profiler

    device = get_device(cfg.training.device)

    print("[Train] Building DataLoaders ...")
    train_loader, val_loader = build_train_val_loaders(
        data_cfg=cfg.data,
        batch_size=cfg.training.batch_size,
    )
    print(f"[Train] Train batches : {len(train_loader)}")
    print(f"[Train] Val batches   : {len(val_loader)}")

    print("[Train] Building model ...")
    model = get_model_on_device(cfg.model, device)
    print(f"[Train] Trainable parameters: {count_parameters(model):,}")

    # Must patch losses AFTER model build, BEFORE Trainer init
    patch_roi_head_losses(model, cfg.loss)

    tb_logger = TensorBoardLogger(
        log_dir=cfg.logging.tensorboard_dir,
        experiment_name=cfg.experiment_name,
    )
    mlf_logger = MLflowLogger(
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        experiment_name=cfg.project_name,
        run_name=cfg.experiment_name,
    )
    mlf_logger.log_params(flat_config_dict(cfg))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        tb_logger=tb_logger,
        mlf_logger=mlf_logger,
    )

    # Resume: explicit path takes priority, then auto-detect latest in exp dir
    resume_path = args.resume
    if resume_path is None:
        exp_ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        latest = get_latest_checkpoint(exp_ckpt_dir)
        if latest:
            print(f"[Train] Auto-resuming from: {latest}")
            resume_path = str(latest)

    if resume_path:
        trainer.resume(resume_path)

    print("[Train] Starting training ...")
    with build_profiler(cfg.profiler):
        trainer.fit()

    tb_logger.close()
    mlf_logger.end_run()

    exp_ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
    best_ckpt = get_latest_checkpoint(exp_ckpt_dir)
    if best_ckpt:
        print(f"[Train] Latest checkpoint: {best_ckpt}")
    return best_ckpt


def _run_inference(args, cfg, ckpt_override: Path = None):
    """
    Run evaluation, visualization, and plotting on val or test split.
    ckpt_override is used when chaining from _run_train.
    """
    from objdet.inference.inference import run_inference
    from objdet.utils.checkpoint import get_latest_checkpoint

    ckpt_path = ckpt_override or args.ckpt
    if ckpt_path is None:
        exp_ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        ckpt_path = get_latest_checkpoint(exp_ckpt_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {exp_ckpt_dir}. "
                "Provide --ckpt explicitly or train first."
            )
        print(f"[Inference] Auto-detected checkpoint: {ckpt_path}")

    output_dir = args.output_dir or f"outputs/inference/{cfg.experiment_name}"

    run_inference(
        config_path=args.config,
        exp_path=args.exp,
        ckpt_path=str(ckpt_path),
        n_samples=args.n_samples,
        score_threshold=args.score_threshold,
        output_dir=output_dir,
        split=args.split,
    )


def _run_export(args, cfg):
    """Export a checkpoint to ONNX."""
    from objdet.models.detector import build_faster_rcnn
    from objdet.onnx.export import export_to_onnx, verify_onnx
    from objdet.utils.checkpoint import load_checkpoint, get_latest_checkpoint
    from objdet.utils.common import get_device

    device = get_device(cfg.training.device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        exp_ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        ckpt_path = get_latest_checkpoint(exp_ckpt_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Provide --ckpt explicitly."
            )
        print(f"[Export] Auto-detected checkpoint: {ckpt_path}")

    onnx_path = args.onnx or f"outputs/model_export/{cfg.experiment_name}.onnx"

    model = build_faster_rcnn(cfg.model)
    load_checkpoint(ckpt_path, model, map_location=str(device))
    export_to_onnx(model, output_path=onnx_path, device=device)
    verify_onnx(onnx_path)


def _save_backbone(args, cfg):
    """Extract and save backbone weights from a full checkpoint."""
    from objdet.models.detector import get_model_on_device
    from objdet.utils.checkpoint import load_checkpoint, save_backbone_only_checkpoint
    from objdet.utils.common import get_device

    device = get_device(cfg.training.device)
    model = get_model_on_device(cfg.model, device)
    load_checkpoint(args.ckpt, model)
    out_path = Path(args.ckpt).parent / "backbone_only.pth"
    save_backbone_only_checkpoint(model, out_path)
    print(f"[SaveBackbone] Saved backbone weights to: {out_path}")


# ===========================================================================
# COSMETIC
# ===========================================================================

def _print_header(cfg, args):
    print(f"\n{'=' * 65}")
    print("  Faster R-CNN — Cityscapes")
    print(f"{'=' * 65}")
    print(f"  Mode        : {args.mode.upper()}")
    print(f"  Experiment  : {cfg.experiment_name}")
    print(f"  Backbone    : {cfg.model.backbone_weights}")
    if args.mode == "train":
        print(f"  Optimizer   : {cfg.training.optimizer}")
        print(f"  Loss cls    : {cfg.loss.classification}")
        print(f"  Loss box    : {cfg.loss.box_regression}")
        print(f"  Epochs      : {cfg.training.epochs}")
        if args.run_inference_after:
            print("  → Inference will run automatically after training")
    if args.mode == "inference":
        print(f"  Checkpoint  : {args.ckpt or 'auto-detect'}")
        print(f"  Split       : {args.split}")
        print(f"  N samples   : {args.n_samples}")
        print(f"  Score thresh: {args.score_threshold}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()