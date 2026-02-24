from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    import yaml

    cfg_path = _repo_root() / "Configs" / "config.yaml"
    if not cfg_path.exists():
        pytest.skip(f"Missing config.yaml at {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _processed_dir(cfg: dict) -> Path:
    return _repo_root() / Path(cfg["data"]["processed_dir"])


def _require_file(p: Path, why: str):
    if not p.exists():
        pytest.skip(f"{why} not found: {p}")


@pytest.mark.data
def test_manifest_has_required_columns():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)

    manifest_path = proc / "manifest.csv"
    _require_file(manifest_path, "manifest")

    df = pd.read_csv(manifest_path)
    required = {"patient_id", "label", "image_dir"}
    missing = required - set(df.columns)
    assert not missing, f"manifest.csv missing columns: {missing}"
    assert len(df) > 0, "manifest.csv is empty"


@pytest.mark.data
def test_splits_exist_and_have_no_patient_leakage():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)

    splits_path = proc / "splits.csv"
    _require_file(splits_path, "splits")

    df = pd.read_csv(splits_path)
    assert {"patient_id", "split"}.issubset(df.columns)

    splits = {}
    for split_name in ["train", "val", "test"]:
        s = set(df.loc[df["split"] == split_name, "patient_id"].astype(str))
        assert len(s) > 0, f"No rows for split={split_name}"
        splits[split_name] = s

    assert splits["train"].isdisjoint(splits["val"]), "patient_id leakage: train vs val"
    assert splits["train"].isdisjoint(splits["test"]), "patient_id leakage: train vs test"
    assert splits["val"].isdisjoint(splits["test"]), "patient_id leakage: val vs test"


@pytest.mark.data
def test_dataset_item_shapes_smoke():
    """
    Loads a single row and checks the dataset returns expected keys and tensor shapes.
    Skips if the referenced DICOM folder doesn't exist locally.
    """
    import torch
    from SRC.dataset import CbisDicomDataset

    cfg = _load_cfg()
    proc = _processed_dir(cfg)

    manifest_path = proc / "manifest.csv"
    _require_file(manifest_path, "manifest")

    df = pd.read_csv(manifest_path)
    row0 = df.iloc[0].copy()

    series_dir = Path(str(row0["image_dir"]))
    if not series_dir.exists():
        pytest.skip(f"Local DICOM folder missing for smoke test: {series_dir}")

    ds = CbisDicomDataset(
        df=df.iloc[:1].copy(),
        img_size=int(cfg["data"]["img_size"]),
        augment=False,
        num_channels=3,
        crop_foreground=True,
    )
    sample = ds[0]
    assert set(sample.keys()) >= {"image", "label", "patient_id"}

    x = sample["image"]
    y = sample["label"]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (3, int(cfg["data"]["img_size"]), int(cfg["data"]["img_size"]))
    assert y.shape == (1,)
    assert torch.isfinite(x).all(), "image tensor contains NaN/Inf"


@pytest.mark.data
def test_eval_outputs_schema_smoke():
    """
    Checks the most recent test_metrics.json has the keys your dashboard expects.
    """
    cfg = _load_cfg()
    proc = _processed_dir(cfg)

    latest_ptr = proc / "latest_run.txt"
    _require_file(latest_ptr, "latest_run.txt")
    run_id = latest_ptr.read_text(encoding="utf-8").strip()
    run_dir = proc / "runs" / run_id
    _require_file(run_dir, "run_dir")

    # Find the newest test_metrics.json under eval_outputs
    metrics_files = sorted(
        run_dir.glob("eval_outputs/**/metrics/test_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not metrics_files:
        pytest.skip(f"No eval_outputs metrics found under {run_dir}")
    metrics_path = metrics_files[0]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Core metrics your pipeline uses everywhere
    for k in ["threshold", "auc", "precision", "recall_sensitivity", "specificity", "tp", "fp", "tn", "fn"]:
        assert k in metrics, f"Missing key in test_metrics.json: {k}"

    # Operating points used by the UI
    assert "operating_points" in metrics and isinstance(metrics["operating_points"], dict)


@pytest.mark.slow
@pytest.mark.data
def test_inference_and_gradcam_shapes_smoke():
    """
    Runs a single inference on one manifest row and checks:
      - probability in [0,1]
      - gradcam shape matches displayed image
      - gradcam is in [0,1]
    Skips if checkpoint or DICOM folder not present locally.
    """
    from SRC.inference import load_config, processed_dir_from_cfg, resolve_latest_run_dir, resolve_checkpoint_path, run_inference

    cfg = load_config()
    proc = processed_dir_from_cfg(cfg)
    run_dir = resolve_latest_run_dir(proc)

    # Need a checkpoint
    try:
        _ = resolve_checkpoint_path(run_dir, "best")
    except FileNotFoundError:
        pytest.skip(f"No checkpoint found in {run_dir} (need model_best.pt)")

    # Need at least one DICOM folder from manifest
    manifest_path = proc / "manifest.csv"
    _require_file(manifest_path, "manifest")
    df = pd.read_csv(manifest_path)

    series_dir = Path(str(df.iloc[0]["image_dir"]))
    if not series_dir.exists():
        pytest.skip(f"Local DICOM folder missing for inference smoke test: {series_dir}")

    out = run_inference(
        series_dir=series_dir,
        ckpt="best",
        threshold=0.5,
        tta=False,
        explain=True,
        prefer_cuda=False,  # keep it predictable for testing
    )

    prob = float(out["prob_malignant"])
    assert 0.0 <= prob <= 1.0

    img = out["img_display"]
    cam = out["gradcam"]

    assert isinstance(img, np.ndarray) and img.ndim == 2
    assert isinstance(cam, np.ndarray) and cam.ndim == 2
    assert img.shape == cam.shape

    assert np.isfinite(cam).all()
    assert cam.min() >= -1e-6
    assert cam.max() <= 1.0 + 1e-6