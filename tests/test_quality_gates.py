from __future__ import annotations

from pathlib import Path
import json
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


def _find_latest_test_metrics(run_dir: Path) -> Path | None:
    candidates = sorted(
        run_dir.glob("eval_outputs/**/metrics/test_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@pytest.mark.data
@pytest.mark.quality
def test_quality_gates_not_terrible():
    """
    These are floors, not targets.
    They should pass on your stable baseline run, and act as regression guards.
    """
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    tm = _find_latest_test_metrics(run_dir)
    if tm is None:
        pytest.skip("No test_metrics.json found under eval_outputs/**/metrics")

    m = json.loads(tm.read_text(encoding="utf-8"))

    auc = float(m.get("auc", -1))
    ap = float(m.get("avg_precision", -1))
    spec = float(m.get("specificity", -1))
    sens = float(m.get("recall_sensitivity", -1))
    brier = float(m.get("brier_score", 999))

    # Conservative floors (shouldn’t be brittle)
    assert auc >= 0.70, f"AUC dropped too low: {auc}"
    assert ap >= 0.50, f"Average precision dropped too low: {ap}"
    assert spec >= 0.85, f"Specificity dropped too low: {spec}"
    assert sens >= 0.20, f"Sensitivity dropped too low: {sens}"
    assert brier <= 0.35, f"Brier score too high (worse calibration): {brier}"


@pytest.mark.data
@pytest.mark.quality
def test_operating_point_present_and_reasonable():
    cfg = _load_cfg()
    proc = _processed_dir(cfg)
    run_dir = _latest_run_dir(proc)

    tm = sorted(
        run_dir.glob("eval_outputs/**/metrics/test_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not tm:
        pytest.skip("No test_metrics.json found")
    m = json.loads(tm[0].read_text(encoding="utf-8"))

    ops = m.get("operating_points", {})
    assert isinstance(ops, dict) and len(ops) > 0, "operating_points missing/empty"

    # Your chosen point is typically sens_at_spec_0.90
    op = ops.get("sens_at_spec_0.90")
    assert op is not None, "operating_points missing sens_at_spec_0.90"
    assert 0.0 <= float(op.get("threshold", -1)) <= 1.0, "operating point threshold not in [0,1]"
    assert 0.0 <= float(op.get("specificity", -1)) <= 1.0, "operating point specificity not in [0,1]"
    assert 0.0 <= float(op.get("sensitivity", -1)) <= 1.0, "operating point sensitivity not in [0,1]"