import os
from pathlib import Path
import logging

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# -------------------- PROJECT NAME --------------------
project_name = "objdet"

# -------------------- FILE STRUCTURE --------------------
list_of_files = [

    # root
    "main.py",
    "requirements.txt",
    "Dockerfile",

    # config
    "config/config.yaml",
    "config/experiments/exp_01.yaml",

    # src structure
    f"src/{project_name}/constants/__init__.py",

    f"src/{project_name}/entity/config_entity.py",

    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/datasets/cityscapes_dataset.py",
    f"src/{project_name}/datasets/transforms.py",

    f"src/{project_name}/models/backbone.py",
    f"src/{project_name}/models/neck.py",
    f"src/{project_name}/models/detector.py",

    f"src/{project_name}/losses/losses.py",

    f"src/{project_name}/evaluation/metrics.py",

    f"src/{project_name}/training/trainer.py",

    f"src/{project_name}/tracking/mlflow_logger.py",
    f"src/{project_name}/tracking/tensorboard_logger.py",

    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/checkpoint.py",
    f"src/{project_name}/utils/visualization.py",
]

# -------------------- FOLDER STRUCTURE --------------------
folders = [
    "data",

    f"src/{project_name}/constants",
    f"src/{project_name}/entity",
    f"src/{project_name}/config",
    f"src/{project_name}/datasets",
    f"src/{project_name}/models",
    f"src/{project_name}/losses",
    f"src/{project_name}/evaluation",
    f"src/{project_name}/training",
    f"src/{project_name}/tracking",
    f"src/{project_name}/utils",

    "config/experiments",

    "outputs/checkpoints",
    "outputs/logs",
    "outputs/tensorboard",
    "outputs/mlruns",
    "outputs/predictions",
    "outputs/profiler",

    "research"
]

# -------------------- CREATE FOLDERS --------------------
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Created folder: {folder}")

# -------------------- CREATE FILES --------------------
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # create file only if not exists or empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath} — skipping")

print("\n✅ Project structure created successfully!\n")