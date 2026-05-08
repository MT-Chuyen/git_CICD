#!/usr/bin/env python3
"""Sanity-check the dataset described by data.yaml.

Exits non-zero on any problem so CI fails loudly. Verifies:
  * data.yaml exists and references existing image/label dirs
  * each image has a matching label file (empty allowed = background)
  * every label line has 5 floats: class cx cy w h
  * class IDs are within [0, nc-1]
  * cx, cy, w, h are within [0, 1]
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def fail(msg: str, errors: list):
    errors.append(msg)


def check_split(dataset_dir: Path, split: str, nc: int, errors: list) -> int:
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"

    if not img_dir.is_dir():
        fail(f"[{split}] missing images dir: {img_dir}", errors)
        return 0
    if not lbl_dir.is_dir():
        fail(f"[{split}] missing labels dir: {lbl_dir}", errors)
        return 0

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        fail(f"[{split}] no images found in {img_dir}", errors)
        return 0

    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            fail(f"[{split}] missing label for {img.name}", errors)
            continue
        for ln, line in enumerate(lbl.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                fail(f"[{split}] {lbl.name}:{ln} expected 5 fields, got {len(parts)}", errors)
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                fail(f"[{split}] {lbl.name}:{ln} non-numeric values", errors)
                continue
            if not 0 <= cid < nc:
                fail(f"[{split}] {lbl.name}:{ln} class id {cid} not in [0,{nc-1}]", errors)
            for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                if not 0.0 <= val <= 1.0:
                    fail(f"[{split}] {lbl.name}:{ln} {name}={val} not in [0,1]", errors)
    return len(images)


def main() -> int:
    errors: list[str] = []

    if not DATA_YAML.is_file():
        print(f"data.yaml not found at {DATA_YAML}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(DATA_YAML.read_text())
    for key in ("path", "train", "val", "nc", "names"):
        if key not in cfg:
            fail(f"data.yaml missing key: {key}", errors)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    nc = int(cfg["nc"])
    if len(cfg["names"]) != nc:
        fail(f"nc={nc} but names has {len(cfg['names'])} entries", errors)

    dataset_dir = (DATA_YAML.parent / cfg["path"]).resolve()
    if not dataset_dir.is_dir():
        fail(f"dataset path does not exist: {dataset_dir}", errors)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    totals = {s: check_split(dataset_dir, s, nc, errors) for s in ("train", "val", "test")}

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\nFAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1

    print("Dataset OK")
    for s, n in totals.items():
        print(f"  {s}: {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
