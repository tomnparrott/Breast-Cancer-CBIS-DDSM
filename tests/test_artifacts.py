from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest


# Resolve all test paths from the repository root
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Load the shared config that tells the tests where processed outputs live
def _load_cfg() -> dict:
    import yaml

    cfg_path = _repo_root() / "Configs" / "config.yaml"
    if not cfg_path.exists():
        pytest.skip(f"Missing config.yaml at {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# Resolve the processed data directory from config
def _processed_dir(cfg: dict) -> Path:
    # Works for both relative and absolute paths
    return _repo_root() / Path(cfg["data"]["processed_dir"])


# Skip the test when a required local artifact is not present
def _require_file(p: Path, why: str):
    if not p.exists():
        pytest.skip(f"{why} not found: {p}")


# Resolve the latest recorded training run directory
def _latest_run_dir(proc: Path) -> Path:
    latest_ptr = proc / "latest_run.txt"
    _require_file(latest_ptr, "latest_run.txt")
    run_id = latest_ptr.read_text(encoding="utf-8").strip()
    run_dir = proc / "runs" / run_id
    _require_file(run_dir, "latest run directory")
    return run_dir


# Find the newest metrics folder created by evaluation
def _find_latest_metrics_dir(run_dir: Path) -> Path | None:
    # finds .../eval_outputs/**/metrics
    candidates = sorted(
        run_dir.glob("eval_outputs/**/metrics"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# Locate `case_index.csv` whether it was saved at run level or metrics level
def _find_case_index(run_dir: Path, metrics_dir: Path | None) -> Path | None:
    p1 = run_dir / "case_index.csv"
    if p1.exists():
        return p1
    if metrics_dir is not None:
        p2 = metrics_dir / "case_index.csv"
        if p2.exists():
            return p2
    return None


# Confirm the latest run kept both best and final checkpoints
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


# Check run metadata exists and still includes the core provenance fields
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


# Verify the latest evaluation produced the expected core artifact files
@pytest.mark.data
@pytest.mark.artifacts
def test_eval_outputs_exist_and_have_expected_files():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    expected = [
        metrics_dir / "test_metrics.json",
        metrics_dir / "test_metrics.csv",
        metrics_dir / "test_preds.npz",
        metrics_dir / "threshold_sweep.csv",
        metrics_dir / "subgroup_metrics.csv",
        metrics_dir / "bootstrap_cis.csv",
        metrics_dir / "calibration_bins.csv",
        metrics_dir / "failure_cases.csv",
        metrics_dir / "test_predictions_with_meta.csv",
        metrics_dir / "eval_pack_manifest.json",
        # Audit outputs
        metrics_dir / "audit_summary.csv",
        metrics_dir / "audit_mask_ablation.csv",
        metrics_dir / "audit_mask_ablation_summary.csv",
        # Patch C outputs
        metrics_dir / "density_policy_thresholds.csv",
        metrics_dir / "density_policy_test_results.csv",
    ]

    missing = [p.name for p in expected if not p.exists()]
    assert not missing, f"Missing eval artifact(s): {missing}"

    # Case index should exist either in run dir or metrics dir after running SRC.case_index
    case_index_path = _find_case_index(run_dir, metrics_dir)
    assert case_index_path is not None and case_index_path.exists(), "case_index.csv missing (run: py -m SRC.case_index)"


# Check the standard evaluation figures were generated
@pytest.mark.data
@pytest.mark.artifacts
def test_figures_exist():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    figures_dir = metrics_dir.parent / "figures"
    assert figures_dir.exists(), f"Figures folder missing at {figures_dir} (run: py -m SRC.eval)"

    expected = [
        figures_dir / "roc_curve.png",
        figures_dir / "pr_curve.png",
        figures_dir / "confusion_matrix.png",
        figures_dir / "calibration_curve.png",
    ]
    missing = [p.name for p in expected if not p.exists()]
    assert not missing, f"Missing figure(s): {missing}"


# Validate the audit ablation CSV includes the columns used by analysis code
@pytest.mark.data
@pytest.mark.artifacts
def test_audit_artifacts_schema():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    ab_path = metrics_dir / "audit_mask_ablation.csv"
    _require_file(ab_path, "audit_mask_ablation.csv")

    ab = pd.read_csv(ab_path)
    if len(ab) == 0:
        pytest.skip("audit_mask_ablation.csv is empty (audit disabled or no cases processed)")

    required_cols = {
        "row_index",
        "y_true",
        "y_pred",
        "outcome",
        "p_orig",
        "p_masked",
        "delta_masked",
        "audit_cam_outside_ratio",
        "audit_cam_edge_ratio",
        "audit_flag_any",
    }
    assert required_cols.issubset(set(ab.columns)), f"Missing cols: {required_cols - set(ab.columns)}"

### Do i need to do @pytest.mark.data for all three functions? Circle back to this

# Validate the density-policy CSV outputs before downstream reporting uses them
@pytest.mark.data
@pytest.mark.artifacts
def test_density_policy_artifacts_schema():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    thr_path = metrics_dir / "density_policy_thresholds.csv"
    res_path = metrics_dir / "density_policy_test_results.csv"
    _require_file(thr_path, "density_policy_thresholds.csv")
    _require_file(res_path, "density_policy_test_results.csv")

    thr = pd.read_csv(thr_path)
    required_thr = {"density_group", "target_specificity", "threshold", "n", "pos", "neg", "status"}
    assert required_thr.issubset(set(thr.columns)), f"Missing cols in density_policy_thresholds.csv: {required_thr - set(thr.columns)}"
    assert len(thr) > 0, "density_policy_thresholds.csv is empty"

    res = pd.read_csv(res_path)
    required_res = {"group_by", "group", "n", "pos", "neg", "sensitivity", "specificity", "tp", "fp", "tn", "fn"}
    assert required_res.issubset(set(res.columns)), f"Missing cols in density_policy_test_results.csv: {required_res - set(res.columns)}"
    assert (res["group_by"] == "overall").any(), "density_policy_test_results.csv missing overall row"

# Confirm prediction metadata includes the density-policy columns added by evaluation
@pytest.mark.data
@pytest.mark.artifacts
def test_test_predictions_with_meta_has_density_policy_columns():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    if metrics_dir is None:
        pytest.skip(f"No eval_outputs/**/metrics found under {run_dir}")

    p = metrics_dir / "test_predictions_with_meta.csv"
    _require_file(p, "test_predictions_with_meta.csv")

    df = pd.read_csv(p)

    required = {"threshold_density_policy", "y_pred_density_policy"}
    missing = required - set(df.columns)
    assert not missing, f"Missing density-policy columns in test_predictions_with_meta.csv: {missing}"

    # light sanity checks (avoid brittle assertions)
    assert df["threshold_density_policy"].notna().any(), "threshold_density_policy is all NaN"
    assert set(df["y_pred_density_policy"].dropna().astype(int).unique()).issubset({0, 1}), "y_pred_density_policy not binary"

# Check the case index keeps the fields needed for case review and ranking
@pytest.mark.data
@pytest.mark.artifacts
def test_case_index_schema_and_sanity():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    metrics_dir = _find_latest_metrics_dir(run_dir)
    case_index_path = _find_case_index(run_dir, metrics_dir)
    if case_index_path is None:
        pytest.skip("No case_index.csv found (run: py -m SRC.case_index)")

    df = pd.read_csv(case_index_path)

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

    assert df["y_prob"].notna().all(), "y_prob has NaNs"
    assert ((df["y_prob"] >= 0) & (df["y_prob"] <= 1)).all(), "y_prob out of [0,1] range"

    assert set(df["y_true"].unique()).issubset({0, 1}), "y_true contains non-binary values"
    assert set(df["y_pred"].unique()).issubset({0, 1}), "y_pred contains non-binary values"
