from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest


# Resolve project-relative paths from the repo root
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Load the shared pipeline config used to find processed outputs
def _load_cfg() -> dict:
    import yaml

    cfg_path = _repo_root() / "Configs" / "config.yaml"
    if not cfg_path.exists():
        pytest.skip(f"Missing config.yaml at {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# Find the processed data directory defined in the config
def _processed_dir(cfg: dict) -> Path:
    return _repo_root() / Path(cfg["data"]["processed_dir"])


# Skip gracefully if a required file or folder is not available locally
def _require_file(p: Path, why: str):
    if not p.exists():
        pytest.skip(f"{why} not found: {p}")


# Resolve the most recent run recorded by the pipeline
def _latest_run_dir(proc: Path) -> Path:
    latest_ptr = proc / "latest_run.txt"
    _require_file(latest_ptr, "latest_run.txt")
    run_id = latest_ptr.read_text(encoding="utf-8").strip()
    run_dir = proc / "runs" / run_id
    _require_file(run_dir, "latest run directory")
    return run_dir


# Re-run one indexed case and compare it with the saved probability output
@pytest.mark.slow
@pytest.mark.data
def test_inference_probability_matches_case_index_within_tolerance():
    """
    Picks one case from case_index.csv and re-runs inference on CPU.
    Confirms probability roughly matches saved y_prob.
    """
    from SRC.inference import run_inference

    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    case_index_path = run_dir / "case_index.csv"
    if not case_index_path.exists():
        metrics_case_index = sorted(
            run_dir.glob("eval_outputs/**/metrics/case_index.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not metrics_case_index:
            pytest.skip("No case_index.csv found")
        case_index_path = metrics_case_index[0]

    df = pd.read_csv(case_index_path)
    if df.empty:
        pytest.skip("case_index.csv is empty")

    # Choose a mid-ranked row (avoid extreme cases)
    row = df.iloc[min(10, len(df) - 1)]
    series_dir = Path(str(row["image_dir"]))
    if not series_dir.exists():
        pytest.skip(f"Series dir not found locally: {series_dir}")

    saved = float(row["y_prob"])

    out = run_inference(
        series_dir=series_dir,
        ckpt="best",
        threshold=0.5,
        tta=False,
        explain=False,
        prefer_cuda=False,  # deterministic & avoids GPU variance
    )
    prob = float(out["prob_malignant"])

    # Allow small drift due to preprocessing differences / CPU ops
    assert abs(prob - saved) <= 0.02, f"Probability mismatch: saved={saved:.4f}, now={prob:.4f}"
