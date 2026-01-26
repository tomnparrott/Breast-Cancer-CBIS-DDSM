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
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true = []
    y_prob = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        logits = logits.view(-1)
        probs = torch.sigmoid(logits)

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


def save_roc_pr_cm(out_dir, y_true, y_prob, threshold=0.5):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC (AUC={auc:.4f})")
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

    # Confusion matrix at the ONE report threshold (spec>=0.90 threshold)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (thr={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def sensitivity_at_specificity(y_true, y_prob, target_spec=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        best = int(np.argmax(spec))
        return float(tpr[best]), float(thresholds[best]), float(spec[best])

    best = valid[np.argmax(tpr[valid])]
    return float(tpr[best]), float(thresholds[best]), float(spec[best])


def save_calibration_curve(out_dir, y_true, y_prob, n_bins=10):
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

    y_true, y_prob = predict_probs(model, test_loader, device=device)

    # Choose ONE report operating point: maximise sensitivity subject to specificity >= 0.90
    sens90, thr90, spec90 = sensitivity_at_specificity(y_true, y_prob, target_spec=0.90)

    # All thresholded metrics + confusion matrix use thr90 for consistency
    metrics = compute_metrics(y_true, y_prob, threshold=thr90)

    # Add threshold-free / probability-quality metrics
    metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Explicitly store the spec>=0.90 operating point details (rubric requirement)
    metrics["sensitivity_at_spec_0.90"] = sens90
    metrics["threshold_at_spec_0.90"] = thr90
    metrics["specificity_achieved_at_spec_0.90"] = spec90

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_dir = make_eval_out_dir(run_dir)

    # --- save plots (ROC/PR threshold-free; CM uses thr90) ---
    save_roc_pr_cm(out_dir, y_true, y_prob, threshold=thr90)
    save_calibration_curve(out_dir, y_true, y_prob, n_bins=10)

    # --- save metrics + preds ---
    (out_dir / "metrics" / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics" / "test_metrics.csv", index=False)

    np.savez_compressed(
        out_dir / "metrics" / "test_preds.npz",
        y_true=y_true,
        y_prob=y_prob,
    )

    print(f"\nSaved metrics: {out_dir / 'metrics' / 'test_metrics.json'}\n")
    print(f"Saved figures: {out_dir / 'figures'}")
    print(f"Saved metrics CSV: {out_dir / 'metrics' / 'test_metrics.csv'}")
    print(f"Saved preds: {out_dir / 'metrics' / 'test_preds.npz'}")


if __name__ == "__main__":
    main()
