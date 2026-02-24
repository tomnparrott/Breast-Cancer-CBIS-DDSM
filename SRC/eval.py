from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json
import yaml
import numpy as np
import pandas as pd
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

from SRC.dataset import CbisDicomDataset
from SRC.model import make_resnet18_binary


def load_config() -> dict:
    cfg_path = Path("Configs/config.yaml")
    return yaml.safe_load(cfg_path.read_text())


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_run_dir(processed_dir: Path) -> Path:
    latest_ptr = processed_dir / "latest_run.txt"
    if latest_ptr.exists():
        run_id = latest_ptr.read_text(encoding="utf-8").strip()
        candidate = processed_dir / "runs" / run_id
        if candidate.exists():
            return candidate
    return processed_dir


def make_eval_out_dir(run_dir: Path) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_name = run_dir.name
    out_dir = run_dir / "eval_outputs" / date_str / run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    return out_dir


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

    y_true = []
    y_prob = []

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

    y_true = np.concatenate(y_true).ravel()
    y_prob = np.concatenate(y_prob).ravel()

    return y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "auc": auc,
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (thr={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.90) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        best = int(np.argmax(spec))
        return float(tpr[best]), float(thresholds[best]), float(spec[best])

    best = valid[np.argmax(tpr[valid])]
    return float(tpr[best]), float(thresholds[best]), float(spec[best])


def specificity_at_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float = 0.90) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        best = int(np.argmax(tpr))
        return float(spec[best]), float(thresholds[best]), float(tpr[best])

    best = valid[np.argmax(spec[valid])]
    return float(spec[best]), float(thresholds[best]), float(tpr[best])


def save_calibration_curve(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve (bins={n_bins})")
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.png", dpi=200)
    plt.close()


def _ensure_col(df: pd.DataFrame, col: str, default: str = "Unknown") -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)


def _add_density_group(df: pd.DataFrame) -> None:
    """
    Creates a binned density group: 1-2, 3-4, Unknown
    (More stable than per-density when groups are small.)
    """
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


def compute_subgroup_metrics(
    df_pred: pd.DataFrame,
    group_col: str,
    threshold: float,
) -> list[dict]:
    """
    Computes subgroup metrics at a FIXED threshold (the global operating threshold).
    Also computes subgroup ROC-AUC (if subgroup contains both classes).
    """
    out: list[dict] = []
    if group_col not in df_pred.columns:
        return out

    for group_value, g in df_pred.groupby(group_col):
        y_true = g["y_true"].to_numpy().astype(int)
        y_prob = g["y_prob"].to_numpy().astype(float)

        m = compute_metrics(y_true, y_prob, threshold=threshold)

        # Add counts
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
                "sensitivity": float(m["recall_sensitivity"]),
                "specificity": float(m["specificity"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )

    return out


def main() -> None:
    cfg = load_config()

    processed_dir = Path(cfg["data"]["processed_dir"])
    run_dir = resolve_run_dir(processed_dir)

    splits_path = run_dir / "splits.csv"
    ckpt_path = run_dir / "model_best.pt"

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. Run manifestprep + split_data first."
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing {ckpt_path}. Train first to create this checkpoint."
        )

    splits = pd.read_csv(splits_path)
    test_df = splits[splits["split"] == "test"].copy()

    if test_df.empty:
        raise RuntimeError("Test split is empty. Check split generation logic.")

    img_size = int(cfg["data"]["img_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    test_ds = CbisDicomDataset(test_df, img_size=img_size, augment=False)

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

    # Eval options (safe defaults if not present in config)
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    tta = bool(eval_cfg.get("tta", False))
    target_spec = float(eval_cfg.get("target_specificity", 0.90))

    y_true, y_prob = predict_probs(model, test_loader, device=device, tta=tta)

    sens_at_spec, thr_at_spec, spec_achieved = sensitivity_at_specificity(
        y_true, y_prob, target_spec=target_spec
    )

    metrics = compute_metrics(y_true, y_prob, threshold=thr_at_spec)

    op_points = {}

    for s in [0.80, 0.85, 0.90, 0.95]:
        sens, thr, spec = sensitivity_at_specificity(y_true, y_prob, target_spec=s)
        op_points[f"sens_at_spec_{s:.2f}"] = {
            "sensitivity": sens,
            "threshold": thr,
            "specificity": spec,
        }

    for r in [0.80, 0.85, 0.90]:
        spec, thr, sens = specificity_at_sensitivity(y_true, y_prob, target_sens=r)
        op_points[f"spec_at_sens_{r:.2f}"] = {
            "specificity": spec,
            "threshold": thr,
            "sensitivity": sens,
        }

    metrics["operating_points"] = op_points

    metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    metrics[f"sensitivity_at_spec_{target_spec:.2f}"] = sens_at_spec
    metrics[f"threshold_at_spec_{target_spec:.2f}"] = thr_at_spec
    metrics[f"specificity_achieved_at_spec_{target_spec:.2f}"] = spec_achieved
    metrics["tta"] = bool(tta)

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_dir = make_eval_out_dir(run_dir)

    # Figures
    save_roc_pr_cm(out_dir, y_true, y_prob, threshold=thr_at_spec)
    save_calibration_curve(out_dir, y_true, y_prob, n_bins=10)

    # Save core metrics
    (out_dir / "metrics" / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics" / "test_metrics.csv", index=False)

    np.savez_compressed(
        out_dir / "metrics" / "test_preds.npz",
        y_true=y_true,
        y_prob=y_prob,
    )

    # ----------------------------
    # Subgroup evaluation
    # ----------------------------
    # Ensure metadata columns exist (new manifest/splits should include them)
    _ensure_col(test_df, "breast_density", default="Unknown")
    _ensure_col(test_df, "abnormality_type", default="unknown")
    _add_density_group(test_df)

    # Sanity: ordering must match loader ordering (shuffle=False)
    if len(test_df) != len(y_true):
        raise RuntimeError(
            f"Length mismatch: test_df={len(test_df)} vs preds={len(y_true)}. "
            "This should not happen with shuffle=False."
        )

    df_pred = test_df.reset_index(drop=True).copy()
    df_pred["y_true"] = y_true.astype(int)
    df_pred["y_prob"] = y_prob.astype(float)
    df_pred["y_pred"] = (df_pred["y_prob"] >= thr_at_spec).astype(int)

    subgroup_rows: list[dict] = []
    subgroup_rows += compute_subgroup_metrics(df_pred, "breast_density", threshold=thr_at_spec)
    subgroup_rows += compute_subgroup_metrics(df_pred, "density_group", threshold=thr_at_spec)
    subgroup_rows += compute_subgroup_metrics(df_pred, "abnormality_type", threshold=thr_at_spec)

    subgroup_path_json = out_dir / "metrics" / "subgroup_metrics.json"
    subgroup_path_csv = out_dir / "metrics" / "subgroup_metrics.csv"

    subgroup_path_json.write_text(json.dumps(subgroup_rows, indent=2), encoding="utf-8")
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path_csv, index=False)

    # Also save per-case table (handy for Streamlit selectors later)
    df_pred.to_csv(out_dir / "metrics" / "test_predictions_with_meta.csv", index=False)

    print(f"\nSaved metrics: {out_dir / 'metrics' / 'test_metrics.json'}")
    print(f"Saved subgroup metrics: {subgroup_path_json}")
    print(f"Saved figures: {out_dir / 'figures'}")
    print(f"Saved preds: {out_dir / 'metrics' / 'test_preds.npz'}")


if __name__ == "__main__":
    main()