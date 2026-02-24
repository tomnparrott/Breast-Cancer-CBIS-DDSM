from __future__ import annotations

from pathlib import Path
import json
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


def _latest_run_dir(proc: Path) -> Path:
    latest_ptr = proc / "latest_run.txt"
    _require_file(latest_ptr, "latest_run.txt")
    run_id = latest_ptr.read_text(encoding="utf-8").strip()
    run_dir = proc / "runs" / run_id
    _require_file(run_dir, "latest run directory")
    return run_dir


def _find_latest_metrics_dir(run_dir: Path) -> Path | None:
    # run_dir/eval_outputs/<date>/<run>/metrics or similar
    candidates = sorted(
        run_dir.glob("eval_outputs/**/metrics"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@pytest.mark.data
@pytest.mark.artifacts
def test_checkpoints_exist():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    best = run_dir / "model_best.pt"
    final = run_dir / "model_final.pt"

    _require_file(best, "model_best.pt")
    _require_file(final, "model_final.pt")


@pytest.mark.data
@pytest.mark.artifacts
def test_run_info_exists_and_has_core_keys():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    p = run_dir / "run_info.json"
    _require_file(p, "run_info.json")

    info = json.loads(p.read_text(encoding="utf-8"))
    for k in ["timestamp_utc", "git_commit", "python", "torch", "seed", "config_path", "config_snapshot"]:
        assert k in info, f"Missing key in run_info.json: {k}"


@pytest.mark.data
@pytest.mark.artifacts
def test_eval_outputs_exist_and_have_expected_files():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    tm = metrics_dir / "test_metrics.json"
    preds = metrics_dir / "test_preds.npz"
    subgroup = metrics_dir / "subgroup_metrics.csv"
    case_index = metrics_dir / "case_index.csv"

    _require_file(tm, "test_metrics.json")
    _require_file(preds, "test_preds.npz")
    _require_file(subgroup, "subgroup_metrics.csv")
    _require_file(case_index, "case_index.csv")


@pytest.mark.data
@pytest.mark.artifacts
def test_figures_exist():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    # figures live alongside metrics (../figures)
    figures_dir = metrics_dir.parent.parent / "figures"
    if not figures_dir.exists():
        pytest.skip(f"No figures folder found at {figures_dir}")

    expected = [
        figures_dir / "roc_curve.png",
        figures_dir / "pr_curve.png",
        figures_dir / "confusion_matrix.png",
        figures_dir / "calibration_curve.png",
    ]
    missing = [p.name for p in expected if not p.exists()]
    assert not missing, f"Missing figure(s): {missing}"


@pytest.mark.data
@pytest.mark.artifacts
def test_case_index_schema_and_sanity():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    # Prefer run_dir/case_index.csv if present (copied there by case_index step)
    p = run_dir / "case_index.csv"
    if not p.exists():
        metrics_dir = _find_latest_metrics_dir(run_dir)
        if metrics_dir is None:
            pytest.skip("No case_index.csv found in run dir or metrics dir")
        p = metrics_dir / "case_index.csv"
        _require_file(p, "case_index.csv")

    df = pd.read_csv(p)

    required = {
        "image_dir",
        "y_prob",
        "y_pred",
        "y_true",
        "outcome",
        "risk_rank",
        "threshold_used",
    }
    missing = required - set(df.columns)
    assert not missing, f"case_index.csv missing columns: {missing}"

    # Probabilities in [0,1] and finite
    assert df["y_prob"].notna().all(), "y_prob has NaNs"
    assert ((df["y_prob"] >= 0) & (df["y_prob"] <= 1)).all(), "y_prob out of [0,1] range"

    # Labels are binary
    assert set(df["y_true"].unique()).issubset({0, 1}), "y_true contains non-binary values"
    assert set(df["y_pred"].unique()).issubset({0, 1}), "y_pred contains non-binary values"