"""
main.py — Updated entry point

New features:
  - debug mode (--debug or config debug.enabled: true)
  - loss patching integrated
  - experiment-isolated outputs
  - backbone-only checkpoint export (--save-backbone)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from objdet.config.configuration import ConfigurationManager
from objdet.datasets.dataloader import build_train_val_loaders
from objdet.models.detector import get_model_on_device
from objdet.losses.losses import patch_roi_head_losses
from objdet.training.trainer import Trainer
from objdet.tracking.tensorboard_logger import TensorBoardLogger
from objdet.tracking.mlflow_logger import MLflowLogger
from objdet.utils.common import set_seed, get_device, count_parameters, flat_config_dict
from objdet.utils.checkpoint import (
    get_latest_checkpoint, load_checkpoint, save_backbone_only_checkpoint
)
from objdet.profiler.profiler_utils import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN — Cityscapes")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--exp",    default=None,
                        help="Experiment override YAML")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--export", default=None,
                        help="Checkpoint to load for ONNX export")
    parser.add_argument("--onnx",   default="outputs/model_export/faster_rcnn.onnx")
    parser.add_argument("--debug",  action="store_true",
                        help="Run tensor-flow debug and exit (no training)")
    parser.add_argument("--save-backbone", default=None,
                        help="Load this checkpoint and save its backbone only")
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Config
    cfg = ConfigurationManager(
        base_config_path=args.config,
        experiment_config_path=args.exp,
    ).get_config()

    set_seed(args.seed)
    device = get_device(cfg.training.device)

    print(f"\n[Main] Experiment  : {cfg.experiment_name}")
    print(f"[Main] Project     : {cfg.project_name}")
    print(f"[Main] Backbone wts: {cfg.model.backbone_weights}")
    print(f"[Main] Optimizer   : {cfg.training.optimizer}")
    print(f"[Main] Loss cls    : {cfg.loss.classification}")
    print(f"[Main] Loss box    : {cfg.loss.box_regression}\n")

    # 2. Debug mode — print tensor flow shapes and exit
    if args.debug or cfg.debug.enabled:
        from objdet.models.detector import debug_detector
        debug_detector(
            image_height=cfg.debug.image_height,
            image_width=cfg.debug.image_width,
            batch_size=cfg.debug.batch_size,
        )
        return

    # 3. ONNX export mode
    if args.export:
        _run_export(args, cfg, device)
        return

    # 4. Backbone-only checkpoint extraction
    if args.save_backbone:
        model = get_model_on_device(cfg.model, device)
        load_checkpoint(args.save_backbone, model)
        out_path = Path(args.save_backbone).parent / "backbone_only.pth"
        save_backbone_only_checkpoint(model, out_path)
        return

    # 5. Data
    print("[Main] Building DataLoaders ...")
    train_loader, val_loader = build_train_val_loaders(
        data_cfg=cfg.data,
        batch_size=cfg.training.batch_size,
    )
    print(f"[Main] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 6. Model
    print("[Main] Building model ...")
    model = get_model_on_device(cfg.model, device)
    print(f"[Main] Trainable parameters: {count_parameters(model):,}")

    # 7. Patch losses (must happen AFTER model is built, BEFORE Trainer)
    patch_roi_head_losses(model, cfg.loss)

    # 8. Loggers
    # Experiment-isolated TensorBoard: outputs/tensorboard/exp_01/
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

    # 9. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        tb_logger=tb_logger,
        mlf_logger=mlf_logger,
    )

    # Auto-resume from latest checkpoint in experiment folder
    resume_path = args.resume
    if resume_path is None:
        exp_ckpt_dir = Path(cfg.checkpointing.save_dir) / cfg.experiment_name
        latest = get_latest_checkpoint(exp_ckpt_dir)
        if latest:
            print(f"[Main] Auto-resuming from: {latest}")
            resume_path = str(latest)

    if resume_path:
        trainer.resume(resume_path)

    # 10. Train
    print("[Main] Starting training ...")
    with build_profiler(cfg.profiler):
        trainer.fit()

    tb_logger.close()
    mlf_logger.end_run()
    print("[Main] Done.")


def _run_export(args, cfg, device):
    from objdet.models.detector import build_faster_rcnn
    from objdet.onnx.export import export_to_onnx, verify_onnx
    print(f"[Main] Loading checkpoint for ONNX export: {args.export}")
    model = build_faster_rcnn(cfg.model)
    load_checkpoint(args.export, model, map_location=str(device))
    export_to_onnx(model, output_path=args.onnx, device=device)
    verify_onnx(args.onnx)


if __name__ == "__main__":
    main()