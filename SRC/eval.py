from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

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


def main() -> None:
    cfg = load_config()

    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_path = processed_dir / "splits.csv"
    ckpt_path = processed_dir / "model_best.pt"

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
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


    out_json = processed_dir / "test_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics: {out_json}\n")


if __name__ == "__main__":
    main()
