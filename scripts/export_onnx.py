#!/usr/bin/env python3
"""Export a trained YOLO weights file to ONNX.

Reads weights path from $WEIGHTS (default: best.pt in latest run).
"""

import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def latest_best() -> Path | None:
    runs = ROOT / "runs" / "detect"
    if not runs.is_dir():
        return None
    candidates = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> int:
    weights = os.getenv("WEIGHTS")
    weights_path = Path(weights) if weights else latest_best()
    if not weights_path or not weights_path.is_file():
        print("No weights file found. Set $WEIGHTS or train first.", file=sys.stderr)
        return 1

    print(f"Exporting {weights_path} -> ONNX")
    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=int(os.getenv("IMGSZ", "640")), opset=12)
    return 0


if __name__ == "__main__":
    sys.exit(main())
