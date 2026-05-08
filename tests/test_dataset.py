"""Smoke-test the bundled sample dataset and the validator script."""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data.yaml"


def test_data_yaml_loads():
    cfg = yaml.safe_load(DATA_YAML.read_text())
    assert cfg["nc"] == len(cfg["names"])
    for key in ("path", "train", "val", "nc", "names"):
        assert key in cfg


def test_dataset_paths_exist():
    cfg = yaml.safe_load(DATA_YAML.read_text())
    base = (DATA_YAML.parent / cfg["path"]).resolve()
    for split in ("train", "val", "test"):
        assert (base / split / "images").is_dir(), f"missing {split}/images"
        assert (base / split / "labels").is_dir(), f"missing {split}/labels"


def test_validator_passes_on_sample():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_dataset.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"validator failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_every_class_appears_in_train():
    """Catch silent regressions where the sample dataset loses coverage of a class."""
    cfg = yaml.safe_load(DATA_YAML.read_text())
    base = (DATA_YAML.parent / cfg["path"]).resolve()
    seen = set()
    for lbl in (base / "train" / "labels").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if line:
                seen.add(int(line.split()[0]))
    expected = set(cfg["names"].keys())
    assert seen == expected, f"train split missing classes: {expected - seen}"
