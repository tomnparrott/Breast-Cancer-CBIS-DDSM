from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Iterable, Any

import json
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy import ndimage as ndi

from SRC.dataset import CbisDicomDataset
from SRC.model import make_resnet18_binary

# Reuse the exact Grad-CAM + preprocessing used by the Streamlit app
from SRC.inference import preprocess_series_dir, gradcam_resnet18, predict_proba


# Evaluate the latest trained run and export metrics, figures, and audit artifacts
# -----------------------------
# Config / device / paths
# -----------------------------
# Load the shared config that drives evaluation settings and paths
def load_config() -> dict:
    cfg_path = Path("Configs/config.yaml")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# Choose the best available device for batched evaluation
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_run_dir(processed_dir: Path) -> Path:
    """
    Mirrors train.py behaviour:
      processed_dir/latest_run.txt -> run_id -> processed_dir/runs/run_id
    Falls back to processed_dir if pointer missing.
    """
    latest_ptr = processed_dir / "latest_run.txt"
    if latest_ptr.exists():
        run_id = latest_ptr.read_text(encoding="utf-8").strip()
        candidate = processed_dir / "runs" / run_id
        if candidate.exists():
            return candidate
    return processed_dir


def make_eval_out_dir(run_dir: Path) -> Path:
    """
    Output structure:
      run_dir/eval_outputs/YYYY-MM-DD/<run_name>/
        figures/
        metrics/
        failure_cases/
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_name = run_dir.name
    out_dir = run_dir / "eval_outputs" / date_str / run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_cases").mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# Core inference for split
# -----------------------------
@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns y_true, y_prob.
    Optional simple TTA: average prediction with horizontal flip.
    """
    model.eval()

    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []

    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32)

        logits = model(x).view(-1)
        probs = torch.sigmoid(logits)

        if tta:
            x_flip = torch.flip(x, dims=[3])  # flip width
            logits_f = model(x_flip).view(-1)
            probs_f = torch.sigmoid(logits_f)
            probs = 0.5 * (probs + probs_f)

        y_true.append(y.view(-1).cpu().numpy())
        y_prob.append(probs.view(-1).cpu().numpy())

    yt = np.concatenate(y_true).ravel()
    yp = np.concatenate(y_prob).ravel()
    return yt, yp


# Compute the main threshold-based metrics from probabilities
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cmx = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cmx.ravel()

    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "auc": auc,
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# Compute the same core metrics when predictions are already binary
def compute_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()

    cmx = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cmx.ravel()

    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
        "pos": int((y_true == 1).sum()),
        "neg": int((y_true == 0).sum()),
    }


# -----------------------------
# Plots / figures
# -----------------------------
# Save the standard ROC, PR, and confusion-matrix figures for the run
def save_roc_pr_cm(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC (AUC={auc:.4f})" if np.isfinite(auc) else "ROC (AUC=nan)")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=200)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"PR (AP={ap:.4f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", dpi=200)
    plt.close()

    # Confusion matrix at provided threshold
    y_pred = (y_prob >= threshold).astype(int)
    cmx = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=[0, 1])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (thr={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()


# Find the best threshold that reaches a target specificity on the ROC curve
def sensitivity_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.90
) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        best = int(np.argmax(spec))
        return float(tpr[best]), float(thresholds[best]), float(spec[best])

    best = valid[np.argmax(tpr[valid])]
    return float(tpr[best]), float(thresholds[best]), float(spec[best])

# Find the best threshold that reaches a target sensitivity on the ROC curve
def specificity_at_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sens: float = 0.90
) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        best = int(np.argmax(tpr))
        return float(spec[best]), float(thresholds[best]), float(tpr[best])

    best = valid[np.argmax(spec[valid])]
    return float(spec[best]), float(thresholds[best]), float(tpr[best])


# Wrap threshold selection so subgroup calculations can fall back safely
def safe_threshold_for_target_spec(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float,
    fallback_threshold: float,
) -> tuple[float, float, float, str]:
    """
    Returns (threshold, achieved_spec, achieved_sens, status)
    status: "ok" | "single_class" | "no_valid_point" | "error"
    """
    y_true = y_true.astype(int).ravel()
    y_prob = y_prob.astype(float).ravel()

    if len(np.unique(y_true)) < 2:
        return float(fallback_threshold), float("nan"), float("nan"), "single_class"

    try:
        sens, thr, spec = sensitivity_at_specificity(y_true, y_prob, target_spec=target_spec)
        return float(thr), float(spec), float(sens), "ok"
    except Exception:
        return float(fallback_threshold), float("nan"), float("nan"), "error"


# Save the calibration plot that compares predicted risk with observed outcomes
def save_calibration_curve(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve (bins={n_bins})")
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.png", dpi=200)
    plt.close()


# Bin predicted probabilities so calibration error can be inspected numerically
def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = y_true.astype(int).ravel()
    y_prob = y_prob.astype(float).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    rows: list[dict] = []
    n_total = len(y_true)
    for b in range(n_bins):
        m = bin_ids == b
        n = int(m.sum())
        if n == 0:
            rows.append(
                {
                    "bin": b,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "n": 0,
                    "frac": 0.0,
                    "mean_prob": float("nan"),
                    "frac_pos": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue

        mean_prob = float(np.mean(y_prob[m]))
        frac_pos = float(np.mean(y_true[m]))
        rows.append(
            {
                "bin": b,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": n,
                "frac": float(n / max(n_total, 1)),
                "mean_prob": mean_prob,
                "frac_pos": frac_pos,
                "abs_gap": float(abs(frac_pos - mean_prob)),
            }
        )

    return pd.DataFrame(rows)


# Compute expected and maximum calibration error from the calibration bins
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    df_bins = _calibration_bins(y_true, y_prob, n_bins=n_bins)
    valid = df_bins["n"] > 0
    ece = float((df_bins.loc[valid, "frac"] * df_bins.loc[valid, "abs_gap"]).sum())
    mce = float(df_bins.loc[valid, "abs_gap"].max()) if valid.any() else float("nan")
    return ece, mce, df_bins


# -----------------------------
# Threshold sweep
# -----------------------------
# Evaluate the model across many thresholds for later analysis and plotting
def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    rows: list[dict] = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": float(m["recall_sensitivity"]),
                "specificity": float(m["specificity"]),
                "ppv": float(m["ppv"]),
                "npv": float(m["npv"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Bootstrap confidence intervals
# -----------------------------
# Pre-generate bootstrap index samples for repeatable resampling
def _bootstrap_samples(y_true: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = len(y_true)
    return rng.integers(0, n, size=(int(n_boot), n), endpoint=False)


# Convert bootstrap samples into a percentile confidence interval
def _ci_from_samples(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


# Bootstrap the key metrics so the report can show uncertainty ranges
def bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    label: str,
    group_by: Optional[str] = None,
    group: Optional[str] = None,
) -> list[dict]:
    y_true = y_true.astype(int).ravel()
    y_prob = y_prob.astype(float).ravel()

    full_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    full_ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    full_m = compute_metrics(y_true, y_prob, threshold=threshold)
    full_sens = float(full_m["recall_sensitivity"])
    full_spec = float(full_m["specificity"])

    idx = _bootstrap_samples(y_true, n_boot=n_boot, seed=seed)

    auc_vals = []
    ap_vals = []
    sens_vals = []
    spec_vals = []

    for b in range(idx.shape[0]):
        yt = y_true[idx[b]]
        yp = y_prob[idx[b]]

        if len(np.unique(yt)) > 1:
            auc_vals.append(float(roc_auc_score(yt, yp)))
            ap_vals.append(float(average_precision_score(yt, yp)))
        else:
            auc_vals.append(float("nan"))
            ap_vals.append(float("nan"))

        mm = compute_metrics(yt, yp, threshold=threshold)
        sens_vals.append(float(mm["recall_sensitivity"]))
        spec_vals.append(float(mm["specificity"]))

    auc_vals = np.array(auc_vals, dtype=float)
    ap_vals = np.array(ap_vals, dtype=float)
    sens_vals = np.array(sens_vals, dtype=float)
    spec_vals = np.array(spec_vals, dtype=float)

    auc_lo, auc_hi = _ci_from_samples(auc_vals)
    ap_lo, ap_hi = _ci_from_samples(ap_vals)
    sens_lo, sens_hi = _ci_from_samples(sens_vals)
    spec_lo, spec_hi = _ci_from_samples(spec_vals)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())

    base = {
        "scope": label,
        "group_by": "" if group_by is None else str(group_by),
        "group": "" if group is None else str(group),
        "n": int(len(y_true)),
        "pos": pos,
        "neg": neg,
        "threshold": float(threshold),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }

    return [
        {**base, "metric": "auc", "estimate": full_auc, "ci_low": auc_lo, "ci_high": auc_hi},
        {**base, "metric": "avg_precision", "estimate": full_ap, "ci_low": ap_lo, "ci_high": ap_hi},
        {**base, "metric": "sensitivity", "estimate": full_sens, "ci_low": sens_lo, "ci_high": sens_hi},
        {**base, "metric": "specificity", "estimate": full_spec, "ci_low": spec_lo, "ci_high": spec_hi},
    ]


# -----------------------------
# Subgroup helpers
# -----------------------------
# Ensure a metadata column exists before grouping or exporting
def _ensure_col(df: pd.DataFrame, col: str, default: str = "Unknown") -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)


# Collapse raw density labels into the grouped buckets used by analysis
def _add_density_group(df: pd.DataFrame) -> None:
    if "breast_density" not in df.columns:
        df["density_group"] = "Unknown"
        return

    def to_group(v: str) -> str:
        v = str(v).strip()
        if v in {"1", "2"}:
            return "1-2"
        if v in {"3", "4"}:
            return "3-4"
        return "Unknown"

    df["density_group"] = df["breast_density"].map(to_group).astype(str)


# Compute the same metrics separately for each metadata subgroup
def compute_subgroup_metrics(df_pred: pd.DataFrame, group_col: str, threshold: float) -> list[dict]:
    out: list[dict] = []
    if group_col not in df_pred.columns:
        return out

    for group_value, g in df_pred.groupby(group_col):
        y_true = g["y_true"].to_numpy().astype(int)
        y_prob = g["y_prob"].to_numpy().astype(float)

        m = compute_metrics(y_true, y_prob, threshold=threshold)

        pos = int((y_true == 1).sum())
        neg = int((y_true == 0).sum())

        out.append(
            {
                "group_by": group_col,
                "group": str(group_value),
                "n": int(len(g)),
                "pos": pos,
                "neg": neg,
                "threshold": float(threshold),
                "auc": float(m["auc"]) if np.isfinite(m["auc"]) else float("nan"),
                "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
                "sensitivity": float(m["recall_sensitivity"]),
                "specificity": float(m["specificity"]),
                "ppv": float(m["ppv"]),
                "npv": float(m["npv"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )

    return out


# -----------------------------
# Explainability audit + ablation summaries
# -----------------------------
# Build a simple breast-region mask from the display image for audit checks
def build_breast_mask(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)

    nz = img[img > 0]
    if nz.size < 50:
        return img > 0

    t = float(np.percentile(nz, 10))
    t = max(0.02, t * 0.5)
    mask = img > t

    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)

    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return (lab == keep)


# Measure how much Grad-CAM attention falls outside or near the breast region
def cam_audit_metrics(cam: np.ndarray, breast_mask: np.ndarray, edge_margin: int = 12) -> dict:
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)

    H, W = cam.shape[:2]
    mask = breast_mask.astype(bool)

    total = float(cam.sum()) + 1e-8

    outside = float((cam * (~mask)).sum()) / total

    e = int(max(1, edge_margin))
    edge_mask = np.zeros((H, W), dtype=bool)
    edge_mask[:e, :] = True
    edge_mask[-e:, :] = True
    edge_mask[:, :e] = True
    edge_mask[:, -e:] = True
    edge = float((cam * edge_mask).sum()) / total

    dil = ndi.binary_dilation(mask, iterations=3)
    ero = ndi.binary_erosion(mask, iterations=3)
    ring = dil ^ ero
    ring_ratio = float((cam * ring).sum()) / total

    breast_area = float(mask.mean())

    flag_outside = outside >= 0.35
    flag_edge = edge >= 0.25
    flag_ring = ring_ratio >= 0.30
    flag_any = bool(flag_outside or flag_edge or flag_ring)

    return {
        "audit_cam_outside_ratio": outside,
        "audit_cam_edge_ratio": edge,
        "audit_cam_ring_ratio": ring_ratio,
        "audit_breast_area_ratio": breast_area,
        "audit_flag_outside": int(flag_outside),
        "audit_flag_edge": int(flag_edge),
        "audit_flag_ring": int(flag_ring),
        "audit_flag_any": int(flag_any),
    }


# Clean strings before using them in failure-case folder names
def _safe_stem(s: str, max_len: int = 80) -> str:
    s = str(s)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:max_len] if len(out) > max_len else out


# Save the source image, heatmap, and overlay for one exported failure case
def save_failure_case_images(
    case_dir: Path,
    img_disp: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.35,
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)

    img = np.clip(img_disp.astype(np.float32), 0.0, 1.0)
    camn = np.clip(cam_map.astype(np.float32), 0.0, 1.0)

    rgb_img = np.stack([img, img, img], axis=-1)
    heat_rgb = cm.inferno(camn)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * rgb_img + alpha * heat_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    p_in = case_dir / "input.png"
    p_hm = case_dir / "heatmap.png"
    p_ov = case_dir / "overlay.png"

    plt.imsave(p_in, img, cmap="gray")
    plt.imsave(p_hm, heat_rgb)
    plt.imsave(p_ov, overlay)

    return {"input_png": str(p_in), "heatmap_png": str(p_hm), "overlay_png": str(p_ov)}


# Add TP, FP, TN, and FN labels to the per-case prediction table
def add_outcome_column(df: pd.DataFrame) -> None:
    if "outcome" in df.columns:
        return
    df["outcome"] = "TN"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "outcome"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "outcome"] = "FN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "outcome"] = "TP"


# Summarise explainability audit flags overall and by key groupings
def save_audit_summary(df_pred: pd.DataFrame, metrics_dir: Path) -> Path:
    if "audit_flag_any" not in df_pred.columns:
        out = pd.DataFrame([{"group_by": "overall", "group": "", "n": len(df_pred), "pct_flag_any": np.nan}])
        p_out = metrics_dir / "audit_summary.csv"
        out.to_csv(p_out, index=False)
        return p_out

    add_outcome_column(df_pred)

    def pct_flag(s: pd.Series) -> float:
        if len(s) == 0:
            return 0.0
        return float(s.mean() * 100.0)

    rows = []
    rows.append(
        {"group_by": "overall", "group": "", "n": int(len(df_pred)), "pct_flag_any": pct_flag(df_pred["audit_flag_any"])}
    )

    if "density_group" in df_pred.columns:
        for g, d in df_pred.groupby("density_group"):
            rows.append(
                {"group_by": "density_group", "group": str(g), "n": int(len(d)), "pct_flag_any": pct_flag(d["audit_flag_any"])}
            )

    for g, d in df_pred.groupby("outcome"):
        rows.append(
            {"group_by": "outcome", "group": str(g), "n": int(len(d)), "pct_flag_any": pct_flag(d["audit_flag_any"])}
        )

    out = pd.DataFrame(rows)
    p_out = metrics_dir / "audit_summary.csv"
    out.to_csv(p_out, index=False)
    return p_out


# Save the raw mask-ablation cases and a grouped summary table
def save_mask_ablation_outputs(ablation_rows: list[dict], metrics_dir: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(ablation_rows)
    p_cases = metrics_dir / "audit_mask_ablation.csv"
    df.to_csv(p_cases, index=False)

    if df.empty:
        p_sum = metrics_dir / "audit_mask_ablation_summary.csv"
        pd.DataFrame([{"group_by": "overall", "group": "", "n": 0}]).to_csv(p_sum, index=False)
        return p_cases, p_sum

    def summarize(d: pd.DataFrame, group_by: str, group: str) -> dict:
        delta = d["delta_masked"].astype(float)
        return {
            "group_by": group_by,
            "group": group,
            "n": int(len(d)),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "std_delta": float(delta.std(ddof=0)),
            "pct_drop_ge_0_05": float((delta <= -0.05).mean() * 100.0),
            "pct_drop_ge_0_10": float((delta <= -0.10).mean() * 100.0),
            "pct_increase_ge_0_05": float((delta >= 0.05).mean() * 100.0),
        }

    rows = [summarize(df, "overall", "")]

    if "density_group" in df.columns:
        for g, d in df.groupby("density_group"):
            rows.append(summarize(d, "density_group", str(g)))

    for g, d in df.groupby("outcome"):
        rows.append(summarize(d, "outcome", str(g)))

    df_sum = pd.DataFrame(rows)
    p_sum = metrics_dir / "audit_mask_ablation_summary.csv"
    df_sum.to_csv(p_sum, index=False)
    return p_cases, p_sum


# -----------------------------
# Patch C: Density-aware decision policy
# -----------------------------
# Learn a separate threshold per density group from the validation split
def compute_density_policy_thresholds(
    df_val: pd.DataFrame,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    target_spec: float,
    fallback_threshold: float,
) -> pd.DataFrame:
    dfv = df_val.reset_index(drop=True).copy()
    dfv["y_true"] = y_true_val.astype(int)
    dfv["y_prob"] = y_prob_val.astype(float)

    _ensure_col(dfv, "breast_density", default="Unknown")
    _add_density_group(dfv)

    rows = []
    for g, d in dfv.groupby("density_group"):
        yt = d["y_true"].to_numpy().astype(int)
        yp = d["y_prob"].to_numpy().astype(float)

        thr, spec, sens, status = safe_threshold_for_target_spec(
            yt, yp, target_spec=target_spec, fallback_threshold=fallback_threshold
        )

        rows.append(
            {
                "density_group": str(g),
                "n": int(len(d)),
                "pos": int((yt == 1).sum()),
                "neg": int((yt == 0).sum()),
                "target_specificity": float(target_spec),
                "threshold": float(thr),
                "specificity_achieved": float(spec) if np.isfinite(spec) else np.nan,
                "sensitivity_achieved": float(sens) if np.isfinite(sens) else np.nan,
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values("density_group")


# Apply the learned density thresholds to test predictions and summarise results
def apply_density_policy_to_test(
    df_test_pred: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    fallback_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresh_map = {
        str(r["density_group"]): float(r["threshold"])
        for _, r in thresholds_df.iterrows()
        if pd.notna(r.get("threshold"))
    }

    out = df_test_pred.copy()
    out["threshold_density_policy"] = out["density_group"].astype(str).map(thresh_map).fillna(float(fallback_threshold))
    out["y_pred_density_policy"] = (out["y_prob"].astype(float) >= out["threshold_density_policy"].astype(float)).astype(int)

    # per-group results on TEST under policy
    rows = []
    # overall
    m_all = compute_metrics_from_preds(out["y_true"].to_numpy(), out["y_pred_density_policy"].to_numpy())
    rows.append({"group_by": "overall", "group": "", **m_all})

    for g, d in out.groupby("density_group"):
        m = compute_metrics_from_preds(d["y_true"].to_numpy(), d["y_pred_density_policy"].to_numpy())
        rows.append({"group_by": "density_group", "group": str(g), **m})

    return out, pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
# Run the full evaluation pipeline and write all derived outputs
def main() -> None:
    cfg = load_config()

    processed_dir = Path(cfg["data"]["processed_dir"])
    run_dir = resolve_run_dir(processed_dir)

    splits_path = run_dir / "splits.csv"
    ckpt_path = run_dir / "model_best.pt"

    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}. Run manifestprep + split_data first.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}. Train first to create this checkpoint.")

    splits = pd.read_csv(splits_path)

    # Accept common val naming
    val_df = splits[splits["split"].isin(["val", "valid", "validation"])].copy()
    test_df = splits[splits["split"] == "test"].copy()

    if test_df.empty:
        raise RuntimeError("Test split is empty. Check split generation logic.")
    if val_df.empty:
        raise RuntimeError("Val split is empty or not named 'val'. Required for density-policy (Patch C).")

    img_size = int(cfg["data"]["img_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    seed = int(cfg.get("seed", 42))

    val_ds = CbisDicomDataset(val_df, img_size=img_size, augment=False)
    test_ds = CbisDicomDataset(test_df, img_size=img_size, augment=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = get_device()
    model = make_resnet18_binary(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Eval options
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    tta = bool(eval_cfg.get("tta", False))
    target_spec = float(eval_cfg.get("target_specificity", 0.90))

    ece_bins = int(eval_cfg.get("ece_bins", 10))
    sweep_steps = int(eval_cfg.get("threshold_sweep_steps", 101))
    failure_top_k = int(eval_cfg.get("failure_top_k", 12))

    n_boot_overall = int(eval_cfg.get("bootstrap_n", 1000))
    n_boot_subgroup = int(eval_cfg.get("bootstrap_n_subgroup", min(500, n_boot_overall)))
    boot_seed = int(eval_cfg.get("bootstrap_seed", seed))

    do_audit = bool(eval_cfg.get("explainability_audit", True))
    audit_max_cases = int(eval_cfg.get("audit_max_cases", 0))  # 0 => all

    # ------------------------
    # Predict on TEST
    # ------------------------
    y_true, y_prob = predict_probs(model, test_loader, device=device, tta=tta)
    sens_at_spec, thr_at_spec, spec_achieved = sensitivity_at_specificity(y_true, y_prob, target_spec=target_spec)

    metrics = compute_metrics(y_true, y_prob, threshold=thr_at_spec)
    metrics["threshold"] = float(thr_at_spec)

    # Operating point grid (for tests + report)
    op_points: dict[str, Any] = {}

    for s in [0.80, 0.85, 0.90, 0.95]:
        sens, thr, spec = sensitivity_at_specificity(y_true, y_prob, target_spec=s)
        op_points[f"sens_at_spec_{s:.2f}"] = {
            "sensitivity": float(sens),
            "threshold": float(thr),
            "specificity": float(spec),
        }

    for r in [0.80, 0.85, 0.90]:
        spec, thr, sens = specificity_at_sensitivity(y_true, y_prob, target_sens=r)
        op_points[f"spec_at_sens_{r:.2f}"] = {
            "specificity": float(spec),
            "threshold": float(thr),
            "sensitivity": float(sens),
        }

    metrics["operating_points"] = op_points

    metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    ece, mce, df_bins = compute_ece(y_true, y_prob, n_bins=ece_bins)
    metrics["ece"] = float(ece)
    metrics["mce"] = float(mce)

    metrics[f"sensitivity_at_spec_{target_spec:.2f}"] = float(sens_at_spec)
    metrics[f"threshold_at_spec_{target_spec:.2f}"] = float(thr_at_spec)
    metrics[f"specificity_achieved_at_spec_{target_spec:.2f}"] = float(spec_achieved)
    metrics["tta"] = bool(tta)

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_dir = make_eval_out_dir(run_dir)
    metrics_dir = out_dir / "metrics"
    figs_dir = out_dir / "figures"
    failures_root = out_dir / "failure_cases"

    # Figures (keep calibration curve only)
    save_roc_pr_cm(out_dir, y_true, y_prob, threshold=thr_at_spec)
    save_calibration_curve(out_dir, y_true, y_prob, n_bins=ece_bins)

    # Save core metrics
    (metrics_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(metrics_dir / "test_metrics.csv", index=False)

    np.savez_compressed(metrics_dir / "test_preds.npz", y_true=y_true, y_prob=y_prob)

    # Threshold sweep
    base_thresholds = np.linspace(0.0, 1.0, sweep_steps)
    thresholds = np.unique(np.concatenate([base_thresholds, np.array([thr_at_spec])]))
    df_sweep = threshold_sweep(y_true, y_prob, thresholds=thresholds)

    idx_nearest = int(np.argmin(np.abs(df_sweep["threshold"].to_numpy() - thr_at_spec)))
    df_sweep["is_operating_point"] = 0
    df_sweep.loc[idx_nearest, "is_operating_point"] = 1
    df_sweep.to_csv(metrics_dir / "threshold_sweep.csv", index=False)

    # Calibration bins table (for ECE/MCE evidence)
    df_bins.to_csv(metrics_dir / "calibration_bins.csv", index=False)

    # ----------------------------
    # Build per-case TEST table
    # ----------------------------
    _ensure_col(test_df, "breast_density", default="Unknown")
    _ensure_col(test_df, "abnormality_type", default="unknown")
    _ensure_col(test_df, "laterality", default="Unknown")
    _ensure_col(test_df, "view", default="Unknown")
    _ensure_col(test_df, "abnormality_id", default="Unknown")
    _ensure_col(test_df, "image_dir", default="")
    _add_density_group(test_df)

    if len(test_df) != len(y_true):
        raise RuntimeError(
            f"Length mismatch: test_df={len(test_df)} vs preds={len(y_true)}. "
            "This should not happen with shuffle=False."
        )

    df_pred = test_df.reset_index(drop=True).copy()
    df_pred["y_true"] = y_true.astype(int)
    df_pred["y_prob"] = y_prob.astype(float)
    df_pred["y_pred"] = (df_pred["y_prob"] >= thr_at_spec).astype(int)

    # ----------------------------
    # Subgroup metrics (global thr)
    # ----------------------------
    subgroup_rows: list[dict] = []
    for group_col in ["breast_density", "density_group", "abnormality_type"]:
        subgroup_rows += compute_subgroup_metrics(df_pred, group_col, threshold=thr_at_spec)

    (metrics_dir / "subgroup_metrics.json").write_text(json.dumps(subgroup_rows, indent=2), encoding="utf-8")
    pd.DataFrame(subgroup_rows).to_csv(metrics_dir / "subgroup_metrics.csv", index=False)

    # ----------------------------
    # Bootstrap CIs (overall + subgroups)
    # ----------------------------
    ci_rows: list[dict] = []
    ci_rows += bootstrap_cis(
        y_true=y_true,
        y_prob=y_prob,
        threshold=thr_at_spec,
        n_boot=n_boot_overall,
        seed=boot_seed,
        label="overall",
    )

    for group_col in ["breast_density", "density_group", "abnormality_type"]:
        if group_col not in df_pred.columns:
            continue
        for group_value, g in df_pred.groupby(group_col):
            yt_g = g["y_true"].to_numpy().astype(int)
            yp_g = g["y_prob"].to_numpy().astype(float)
            ci_rows += bootstrap_cis(
                y_true=yt_g,
                y_prob=yp_g,
                threshold=thr_at_spec,
                n_boot=n_boot_subgroup,
                seed=boot_seed,
                label="subgroup",
                group_by=group_col,
                group=str(group_value),
            )

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(metrics_dir / "bootstrap_cis.csv", index=False)
    (metrics_dir / "bootstrap_cis.json").write_text(json.dumps(ci_rows, indent=2), encoding="utf-8")

    # ----------------------------
    # Patch C: Density policy (learn on VAL, apply to TEST)
    # ----------------------------
    y_true_val, y_prob_val = predict_probs(model, val_loader, device=device, tta=tta)

    _ensure_col(val_df, "breast_density", default="Unknown")
    _add_density_group(val_df)

    df_policy_thr = compute_density_policy_thresholds(
        df_val=val_df,
        y_true_val=y_true_val,
        y_prob_val=y_prob_val,
        target_spec=target_spec,
        fallback_threshold=float(thr_at_spec),
    )
    df_policy_thr.to_csv(metrics_dir / "density_policy_thresholds.csv", index=False)

    df_pred, df_policy_results = apply_density_policy_to_test(
        df_test_pred=df_pred,
        thresholds_df=df_policy_thr,
        fallback_threshold=float(thr_at_spec),
    )
    df_policy_results.to_csv(metrics_dir / "density_policy_test_results.csv", index=False)

    # ----------------------------
    # Failure cases + explainability audit + mask ablation
    # ----------------------------
    failure_rows: list[dict] = []

    if do_audit and ("image_dir" in df_pred.columns) and (df_pred["image_dir"].astype(str).str.len().sum() > 0):
        fp = df_pred[(df_pred["y_true"] == 0) & (df_pred["y_pred"] == 1)].copy()
        fn = df_pred[(df_pred["y_true"] == 1) & (df_pred["y_pred"] == 0)].copy()

        fp = fp.sort_values("y_prob", ascending=False).head(max(failure_top_k, 0))
        fn = fn.sort_values("y_prob", ascending=True).head(max(failure_top_k, 0))

        export_idx = set(fp.index.tolist()) | set(fn.index.tolist())

        audit_indices = df_pred.index.to_list()
        if audit_max_cases > 0 and len(audit_indices) > audit_max_cases:
            rng = np.random.default_rng(seed)
            audit_indices = rng.choice(audit_indices, size=audit_max_cases, replace=False).tolist()
            audit_indices.sort()

        ablation_rows: list[dict] = []

        print(f"\nExplainability audit: computing Grad-CAM sanity metrics for {len(audit_indices)} cases...")

        for i in audit_indices:
            row = df_pred.loc[i]
            series_dir = Path(str(row.get("image_dir", "")))
            if not series_dir.exists():
                continue

            try:
                prep = preprocess_series_dir(series_dir=series_dir, img_size=img_size, num_channels=3, crop_foreground=True)
                cam_map = gradcam_resnet18(model, prep.x, device=device, target_layer=None)
                mask = build_breast_mask(prep.img_display)
                audit = cam_audit_metrics(cam_map, mask, edge_margin=12)

                # Mask ablation (breast-only)
                mask_t = torch.from_numpy(mask.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)
                x_orig = prep.x.to(device=device, dtype=torch.float32)
                x_masked = x_orig.clone() * mask_t

                p_masked = predict_proba(model, x_masked, device=device, tta=tta)
                p_orig = float(row["y_prob"])
                delta = float(p_masked - p_orig)

                df_pred.loc[i, "audit_p_masked"] = float(p_masked)
                df_pred.loc[i, "audit_delta_masked"] = float(delta)

                for k, v in audit.items():
                    df_pred.loc[i, k] = v

                outcome = "TN"
                if int(row["y_true"]) == 0 and int(row["y_pred"]) == 1:
                    outcome = "FP"
                elif int(row["y_true"]) == 1 and int(row["y_pred"]) == 0:
                    outcome = "FN"
                elif int(row["y_true"]) == 1 and int(row["y_pred"]) == 1:
                    outcome = "TP"

                ablation_rows.append(
                    {
                        "row_index": int(i),
                        "patient_id": str(row.get("patient_id", "")),
                        "image_dir": str(series_dir),
                        "y_true": int(row["y_true"]),
                        "y_pred": int(row["y_pred"]),
                        "outcome": outcome,
                        "p_orig": p_orig,
                        "p_masked": float(p_masked),
                        "delta_masked": float(delta),
                        "density_group": str(row.get("density_group", "")),
                        "breast_density": str(row.get("breast_density", "")),
                        "abnormality_type": str(row.get("abnormality_type", "")),
                        "view": str(row.get("view", "")),
                        "laterality": str(row.get("laterality", "")),
                        **audit,
                    }
                )

                if i in export_idx:
                    kind = "FP" if int(row["y_true"]) == 0 else "FN"
                    case_key = f"{kind}_idx{i}_pid{_safe_stem(str(row.get('patient_id', '')))}_{_safe_stem(series_dir.name)}"
                    case_dir = failures_root / kind / case_key
                    paths = save_failure_case_images(case_dir, prep.img_display, cam_map, alpha=0.35)

                    failure_rows.append(
                        {
                            "kind": kind,
                            "row_index": int(i),
                            "patient_id": str(row.get("patient_id", "")),
                            "image_dir": str(series_dir),
                            "y_true": int(row["y_true"]),
                            "y_prob": float(row["y_prob"]),
                            "threshold": float(thr_at_spec),
                            **{k: row.get(k, "") for k in ["breast_density", "density_group", "abnormality_type", "view", "laterality", "abnormality_id"]},
                            **paths,
                            **audit,
                        }
                    )

            except Exception as e:
                df_pred.loc[i, "audit_error"] = str(e)

        save_mask_ablation_outputs(ablation_rows, metrics_dir)
        pd.DataFrame(failure_rows).to_csv(metrics_dir / "failure_cases.csv", index=False)

    else:
        pd.DataFrame(failure_rows).to_csv(metrics_dir / "failure_cases.csv", index=False)
        save_mask_ablation_outputs([], metrics_dir)

    # Save per-case table (Streamlit + report references)
    df_pred.to_csv(metrics_dir / "test_predictions_with_meta.csv", index=False)
    save_audit_summary(df_pred, metrics_dir)

    # Manifest
    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figs_dir),
        "failure_cases_dir": str(failures_root),
        "operating_threshold": float(thr_at_spec),
        "target_specificity": float(target_spec),
        "outputs": {
            "core_metrics_json": str(metrics_dir / "test_metrics.json"),
            "threshold_sweep_csv": str(metrics_dir / "threshold_sweep.csv"),
            "subgroup_metrics_csv": str(metrics_dir / "subgroup_metrics.csv"),
            "bootstrap_cis_csv": str(metrics_dir / "bootstrap_cis.csv"),
            "calibration_bins_csv": str(metrics_dir / "calibration_bins.csv"),
            "failure_cases_csv": str(metrics_dir / "failure_cases.csv"),
            "predictions_with_meta_csv": str(metrics_dir / "test_predictions_with_meta.csv"),
            "density_policy_thresholds_csv": str(metrics_dir / "density_policy_thresholds.csv"),
            "density_policy_test_results_csv": str(metrics_dir / "density_policy_test_results.csv"),
            "audit_summary_csv": str(metrics_dir / "audit_summary.csv"),
            "audit_mask_ablation_csv": str(metrics_dir / "audit_mask_ablation.csv"),
            "audit_mask_ablation_summary_csv": str(metrics_dir / "audit_mask_ablation_summary.csv"),
            "fig_roc": str(figs_dir / "roc_curve.png"),
            "fig_pr": str(figs_dir / "pr_curve.png"),
            "fig_cm": str(figs_dir / "confusion_matrix.png"),
            "fig_calibration_curve": str(figs_dir / "calibration_curve.png"),
        },
    }
    (metrics_dir / "eval_pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved metrics:             {metrics_dir / 'test_metrics.json'}")
    print(f"Saved threshold sweep:     {metrics_dir / 'threshold_sweep.csv'}")
    print(f"Saved subgroup metrics:    {metrics_dir / 'subgroup_metrics.csv'}")
    print(f"Saved bootstrap CIs:       {metrics_dir / 'bootstrap_cis.csv'}")
    print(f"Saved calibration bins:    {metrics_dir / 'calibration_bins.csv'}")
    print(f"Saved density thresholds:  {metrics_dir / 'density_policy_thresholds.csv'}")
    print(f"Saved density test res:    {metrics_dir / 'density_policy_test_results.csv'}")
    print(f"Saved failure cases index: {metrics_dir / 'failure_cases.csv'}")
    print(f"Saved figures:             {figs_dir}")
    print(f"Saved preds:               {metrics_dir / 'test_preds.npz'}")


if __name__ == "__main__":
    main()
