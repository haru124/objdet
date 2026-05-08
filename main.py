"""
main.py

Entry point for the Faster R-CNN training pipeline.

Usage:
    # Train with default config
    python main.py

    # Train with experiment override
    python main.py --config config/config.yaml --exp config/experiments/exp_01.yaml

    # Resume from a checkpoint
    python main.py --resume checkpoints/checkpoint_epoch_0004.pth

    # Export trained model to ONNX
    python main.py --export checkpoints/checkpoint_epoch_0020.pth --onnx model.onnx

VSCode debugging tip:
    Add this to .vscode/launch.json:
    {
        "name": "Train",
        "type": "debugpy",
        "request": "launch",
        "program": "main.py",
        "args": ["--exp", "config/experiments/exp_01.yaml"]
    }
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the Python path so `objdet` can be imported
sys.path.insert(0, str(Path(__file__).parent / "src"))

from objdet.config.configuration import ConfigurationManager
from objdet.datasets.dataloader import build_train_val_loaders
from objdet.models.detector import get_model_on_device
from objdet.training.trainer import Trainer
from objdet.tracking.tensorboard_logger import TensorBoardLogger
from objdet.tracking.mlflow_logger import MLflowLogger
from objdet.utils.common import set_seed, get_device, count_parameters, flat_config_dict
from objdet.utils.checkpoint import get_latest_checkpoint, load_checkpoint
#from objdet.profiler.profiler_utils import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN — Cityscapes training")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to base YAML config file")
    parser.add_argument("--exp", default=None,
                        help="Path to experiment override YAML")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--export", default=None,
                        help="Checkpoint to load for ONNX export")
    parser.add_argument("--onnx", default="model_export/faster_rcnn.onnx",
                        help="Output path for ONNX export")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Configuration
    # ------------------------------------------------------------------ #
    cfg = ConfigurationManager(
        base_config_path=args.config,
        experiment_config_path=args.exp,
    ).get_config()

    set_seed(args.seed)
    device = get_device(cfg.training.device)

    # ------------------------------------------------------------------ #
    # 2. ONNX export mode (skip training)
    # ------------------------------------------------------------------ #
    if args.export:
        _run_export(args, cfg, device)
        return

    # ------------------------------------------------------------------ #
    # 3. Data
    # ------------------------------------------------------------------ #
    print("[Main] Building DataLoaders ...")
    train_loader, val_loader = build_train_val_loaders(
        data_cfg=cfg.data,
        batch_size=cfg.training.batch_size,
    )
    print(f"[Main] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------ #
    # 4. Model
    # ------------------------------------------------------------------ #
    print("[Main] Building model ...")
    model = get_model_on_device(cfg.model, device)
    print(f"[Main] Trainable parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------ #
    # 5. Loggers
    # ------------------------------------------------------------------ #
    hparams = flat_config_dict(cfg)

    tb_logger = TensorBoardLogger(
        log_dir=cfg.logging.tensorboard_dir,
        experiment_name=cfg.experiment_name,
    )

    mlf_logger = MLflowLogger(
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        experiment_name=cfg.project_name,
        run_name=cfg.experiment_name,
    )
    mlf_logger.log_params(hparams)

    # ------------------------------------------------------------------ #
    # 6. Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        tb_logger=tb_logger,
        mlf_logger=mlf_logger,
    )

    # Resume from checkpoint if requested (or auto-detect latest)
    resume_path = args.resume or (
        str(get_latest_checkpoint(Path(cfg.checkpointing.save_dir)))
        if get_latest_checkpoint(Path(cfg.checkpointing.save_dir))
        else None
    )
    if resume_path:
        trainer.resume(resume_path)

    # ------------------------------------------------------------------ #
    # 7. Training (with optional profiler)
    # ------------------------------------------------------------------ #
    print("[Main] Starting training ...")
    with build_profiler(cfg.profiler):
        trainer.fit()

    # ------------------------------------------------------------------ #
    # 8. Clean up loggers
    # ------------------------------------------------------------------ #
    tb_logger.close()
    mlf_logger.end_run()
    print("[Main] Done.")


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def _run_export(args, cfg, device):
    from objdet.models.detector import build_faster_rcnn
    from objdet.onnx.export import export_to_onnx, verify_onnx

    print(f"[Main] Loading checkpoint for export: {args.export}")
    model = build_faster_rcnn(cfg.model)
    load_checkpoint(args.export, model, map_location=str(device))
    export_to_onnx(model, output_path=args.onnx, device=device)
    verify_onnx(args.onnx)


if __name__ == "__main__":
    main()