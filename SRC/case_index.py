from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import json

import numpy as np
import pandas as pd

# Resolve paths
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

# Load the config used to find processed data and run outputs
def load_config() -> dict:
    import yaml
    cfg_path = project_root() / "Configs" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# Resolve the latest run directory recorded by training
def resolve_latest_run_dir(processed_dir: Path) -> Path:
    latest_ptr = processed_dir / "latest_run.txt"
    if latest_ptr.exists():
        run_id = latest_ptr.read_text(encoding="utf-8").strip()
        candidate = processed_dir / "runs" / run_id
        if candidate.exists():
            return candidate
    return processed_dir

# Find the newest metrics folder created by the evaluation step
def find_latest_eval_metrics_dir(run_dir: Path) -> Path:
    root = run_dir / "eval_outputs"
    if not root.exists():
        raise FileNotFoundError(f"No eval_outputs folder found in: {run_dir}")

    date_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not date_dirs:
        raise FileNotFoundError(f"No dated eval_outputs found in: {root}")

    date_dirs.sort(key=lambda p: p.name)
    latest_date_dir = date_dirs[-1]

    run_name = run_dir.name
    candidate = latest_date_dir / run_name / "metrics"
    if candidate.exists():
        return candidate

    run_folders = [p for p in latest_date_dir.iterdir() if p.is_dir()]
    if not run_folders:
        raise FileNotFoundError(f"No run folders inside: {latest_date_dir}")

    run_folders.sort(key=lambda p: p.name)
    return run_folders[-1] / "metrics"

# Helper functions #

# Collapse density values into the grouped labels used by the dashboard
def add_density_group(df: pd.DataFrame) -> None:
    if "breast_density" not in df.columns:
        df["density_group"] = "Unknown"
        return

    def to_group(v: object) -> str:
        s = str(v).strip()
        if s in {"1", "2"}:
            return "1-2"
        if s in {"3", "4"}:
            return "3-4"
        return "Unknown"

    df["density_group"] = df["breast_density"].map(to_group).astype(str)

# Convert prediction values into confusion matrix labels
def outcome_label(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "FN"

# Read the evaluation threshold saved with the latest metrics
def load_threshold_from_metrics(metrics_json_path: Path) -> float:
    if not metrics_json_path.exists():
        return 0.5

    m = json.loads(metrics_json_path.read_text(encoding="utf-8"))

    if "threshold" in m:
        try:
            return float(m["threshold"])
        except Exception:
            pass

    for k in ["threshold_at_spec_0.90", "threshold_at_spec_0.90"]:
        if k in m:
            try:
                return float(m[k])
            except Exception:
                pass

    return 0.5

# Core functions

# Build the per case index used by Streamlit for the dashboard
def build_case_index(
    splits_csv: Path,
    metrics_dir: Path,
    threshold_override: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Builds a per-case table for Dashboard:
      - includes metadata
      - attaches y_true/y_prob/y_pred
      - labels TP/FP/TN/FN at the evaluation threshold
      - ranks by risk 
    """
    splits = pd.read_csv(splits_csv)
    test_df = splits[splits["split"] == "test"].copy().reset_index(drop=True)

    pred_with_meta = metrics_dir / "test_predictions_with_meta.csv"
    metrics_json = metrics_dir / "test_metrics.json"
    threshold = float(threshold_override) if threshold_override is not None else load_threshold_from_metrics(metrics_json)

    # Predictions with metadata 
    if pred_with_meta.exists():
        df = pd.read_csv(pred_with_meta)
        # Check columns are present
        if "y_true" not in df.columns or "y_prob" not in df.columns:
            raise RuntimeError(f"{pred_with_meta} exists but is missing y_true/y_prob.")
        if "y_pred" not in df.columns:
            df["y_pred"] = (df["y_prob"].astype(float) >= threshold).astype(int)
    else:
        # If not, use splits test_df + test_preds.npz
        preds_npz = metrics_dir / "test_preds.npz"
        if not preds_npz.exists():
            raise FileNotFoundError(
                "Missing both test_predictions_with_meta.csv and test_preds.npz in metrics dir"
                f"Expected at: {pred_with_meta} or {preds_npz}"
            )
        
        # Load the .npz predictions and attach to test_df
        data = np.load(preds_npz)
        y_true = data["y_true"].astype(int).ravel()
        y_prob = data["y_prob"].astype(float).ravel()

        if len(test_df) != len(y_true):
            raise RuntimeError(
                f"Length mismatch: test_df={len(test_df)} vs preds={len(y_true)}. "
                "Ensure eval was run with shuffle=False and the same splits.csv."
            )

        df = test_df.copy()
        df["y_true"] = y_true
        df["y_prob"] = y_prob
        df["y_pred"] = (df["y_prob"] >= threshold).astype(int)

    # Ensure  meta columns exist
    for col, default in [
        ("breast_density", "Unknown"),
        ("abnormality_type", "unknown"),
        ("laterality", "Unknown"),
        ("view", "Unknown"),
        ("abnormality_id", "Unknown"),
        ("source_csv", "unknown"),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    add_density_group(df)

    # Outcome labels
    df["outcome"] = [
        outcome_label(int(t), int(p)) for t, p in zip(df["y_true"].astype(int), df["y_pred"].astype(int))
    ]

    # Risk rank (1 = highest probability of malignancy)
    df["risk_rank"] = df["y_prob"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # Add the threshold used
    df["threshold_used"] = threshold

    return df, threshold

# Latest run outputs and write the case index files
def main() -> None:
    cfg = load_config()
    processed_dir = project_root() / Path(cfg["data"]["processed_dir"])
    run_dir = resolve_latest_run_dir(processed_dir)

    # Check for splits.csv
    splits_csv = processed_dir / "splits.csv"
    if not splits_csv.exists():
        raise FileNotFoundError(f"Missing splits.csv at: {splits_csv}")

    metrics_dir = find_latest_eval_metrics_dir(run_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df, thr = build_case_index(splits_csv=splits_csv, metrics_dir=metrics_dir, threshold_override=None)

    # Save outputs
    out_csv = metrics_dir / "case_index.csv"
    out_json = metrics_dir / "case_index_summary.json"

    df.to_csv(out_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "metrics_dir": str(metrics_dir),
        "threshold_used": float(thr),
        "n": int(len(df)),
        "counts": df["outcome"].value_counts().to_dict(),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Copy case index to the run directory for easy access by the dashboard
    df.to_csv(run_dir / "case_index.csv", index=False)

    print(f"Saved case index: {out_csv}")
    print(f"Saved summary:    {out_json}")
    print(f"Copied to run:    {run_dir / 'case_index.csv'}")

if __name__ == "__main__":
    main()