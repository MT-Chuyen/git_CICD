# Gate Detection — CI/CD test project

Standalone, self-contained test version of the gate-detection training pipeline used to
prototype CI/CD before wiring it into the main project. Bundles a tiny sample of the real
dataset (~19 images covering all 5 classes) so the smoke training job can run end-to-end.

## Layout

```
.
├── data.yaml                 # points at sample_dataset/ via a relative path
├── train.py                  # full training entrypoint (env-vars override defaults)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml            # ruff + pytest config
├── sample_dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── test/{images,labels}/
├── scripts/
│   ├── validate_dataset.py   # schema + label sanity check, used by CI
│   ├── count_classes.py      # class distribution report
│   └── export_onnx.py        # weights -> ONNX
├── tests/
│   └── test_dataset.py       # pytest wrapper around the validator
└── .github/workflows/ci.yml  # lint + validate + smoke train (1 epoch, CPU)
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Validate the dataset
python scripts/validate_dataset.py

# 2. Smoke-train (1 epoch on the bundled sample)
EPOCHS=1 BATCH=2 IMGSZ=320 WEIGHTS=yolo11n.pt RUN_NAME=local_smoke python train.py

# 3. Full training (override what you need)
EPOCHS=150 BATCH=32 IMGSZ=640 WEIGHTS=yolo11s.pt RUN_NAME=full_run python train.py
```

`train.py` reads `EPOCHS`, `IMGSZ`, `BATCH`, `WEIGHTS`, `RUN_NAME`, `PROJECT_DIR` from the
environment, with the same defaults as the original training script. When the `CI`
environment variable is set, W&B is disabled automatically.

## CI pipeline

The workflow at `.github/workflows/ci.yml` runs on every push/PR and does:

1. `ruff check` — lint
2. `python scripts/validate_dataset.py` — dataset schema + label sanity
3. `pytest -q` — same validation, plus any future tests
4. **Smoke train**: `EPOCHS=1 BATCH=2 IMGSZ=320 WEIGHTS=yolo11n.pt python train.py`
   on the bundled sample, on CPU. Catches broken `data.yaml`, missing labels,
   dependency drift. Takes a few minutes.

Full GPU training is **not** run in CI — wire that up later as a manually-triggered job
on a self-hosted runner once the smoke pipeline is stable.

## Why a separate folder

The parent `Gate_Detection/` directory contains the real dataset, large model weights,
videos, and run logs that should not be committed. This subfolder is the minimal slice
needed to exercise the CI pipeline without dragging any of that in.
