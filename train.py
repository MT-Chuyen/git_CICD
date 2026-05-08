#!/usr/bin/env python3
"""Train YOLO gate detection model.

Classes: Gate_In, Gate_Out, Gate_6, In_Gate_Zone, Gate_Metal

Reads training settings from environment variables so the same script works
both locally (full training) and in CI (1-epoch smoke run).
"""

import os
from pathlib import Path

os.environ.setdefault("WANDB_PROJECT", "gate-detection")
# Keep CI runs offline-friendly: no W&B login prompt.
if os.getenv("CI"):
    os.environ.setdefault("WANDB_MODE", "disabled")

from albumentations import CLAHE, RandomBrightnessContrast, RandomGamma
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = str(ROOT / "data.yaml")

EPOCHS = int(os.getenv("EPOCHS", "150"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
BATCH = int(os.getenv("BATCH", "32"))
WEIGHTS = os.getenv("WEIGHTS", "yolo11s.pt")
RUN_NAME = os.getenv("RUN_NAME", "gate_detect_cicd_test")
PROJECT_DIR = os.getenv("PROJECT_DIR", str(ROOT / "runs" / "detect"))


def main():
    model = YOLO(WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=30,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        hsv_h=0.1,
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.3,
        augmentations=[
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
            CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
        ],
    )


if __name__ == "__main__":
    main()
