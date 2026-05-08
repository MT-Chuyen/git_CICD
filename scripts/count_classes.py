#!/usr/bin/env python3
"""Count instances of each class across train/val/test label folders."""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data.yaml"


def load_config():
    cfg = yaml.safe_load(DATA_YAML.read_text())
    base = (DATA_YAML.parent / cfg["path"]).resolve()
    return base, cfg["names"]


def count_split(dataset_dir: Path, split: str):
    label_dir = dataset_dir / split / "labels"
    counts = defaultdict(int)
    n_images = 0
    n_background = 0
    for f in label_dir.glob("*.txt"):
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        n_images += 1
        if not lines:
            n_background += 1
        for line in lines:
            counts[int(line.split()[0])] += 1
    return counts, n_images, n_background


def main():
    dataset_dir, class_names = load_config()
    splits = ["train", "val", "test"]

    all_counts = {s: count_split(dataset_dir, s) for s in splits}
    all_classes = sorted(class_names.keys())
    col_w, name_w = 14, 16

    header = f"{'Class':<{name_w}}" + "".join(f"{s:>{col_w}}" for s in splits) + f"{'TOTAL':>{col_w}}"
    print(header)
    print("-" * len(header))

    totals = {s: 0 for s in splits}
    for cid in all_classes:
        row = f"{class_names[cid]:<{name_w}}"
        total = 0
        for s in splits:
            n = all_counts[s][0][cid]
            row += f"{n:>{col_w}}"
            total += n
            totals[s] += n
        row += f"{total:>{col_w}}"
        print(row)

    print("-" * len(header))
    print(f"{'Images':<{name_w}}" + "".join(f"{all_counts[s][1]:>{col_w}}" for s in splits))
    print(f"{'Background':<{name_w}}" + "".join(f"{all_counts[s][2]:>{col_w}}" for s in splits))


if __name__ == "__main__":
    sys.exit(main())
