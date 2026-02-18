from pathlib import Path
import yaml
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

import json
import datetime
import platform
import subprocess

from SRC.seed import set_seed
from SRC.model import make_resnet18_binary
from SRC.dataset import CbisDicomDataset
from SRC.split_data import split_by_patient


def load_config() -> dict:
    cfg_path = Path("Configs/config.yaml")
    return yaml.safe_load(cfg_path.read_text())


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def make_bce_loss(cfg: dict, device: str) -> nn.Module:
    return nn.BCEWithLogitsLoss()


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(processed_dir: Path, seed: int) -> Path:
    runs_root = processed_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S") + f"_seed{seed}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # pointer to latest
    (processed_dir / "latest_run.txt").write_text(run_id, encoding="utf-8")
    return run_dir


def write_run_info(run_dir: Path, cfg: dict) -> None:
    cfg_path = Path("Configs/config.yaml")
    cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""

    info = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": get_git_hash(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": int(cfg.get("seed", -1)),
        "config_path": str(cfg_path),
        "config_snapshot": cfg_text,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, cfg: dict) -> dict:
    model.eval()
    criterion = make_bce_loss(cfg, device)

    total_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_probs = []

    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += float(loss.item()) * x.size(0)
        correct += int((preds == y).sum().item())
        total += x.size(0)

        all_targets.extend(y.detach().cpu().numpy().reshape(-1))
        all_probs.extend(probs.detach().cpu().numpy().reshape(-1))

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc": correct / max(total, 1),
        "auc": float(auc),
    }


def main() -> None:
    cfg = load_config()
    seed = int(cfg["seed"])
    set_seed(seed)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(processed_dir, seed)
    write_run_info(run_dir, cfg)

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. "
            "Run Scripts/manifestprep.py after the download completes."
        )

    manifest = pd.read_csv(manifest_path)

    splits = split_by_patient(
        manifest=manifest,
        val_frac=float(cfg["split"]["val_frac"]),
        test_frac=float(cfg["split"]["test_frac"]),
        random_state=seed,
    )

    # Save splits for reproducibility
    splits.to_csv(run_dir / "splits.csv", index=False)
    splits.to_csv(processed_dir / "splits.csv", index=False)

    img_size = int(cfg["data"]["img_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_df = splits[splits["split"] == "train"].copy()
    val_df = splits[splits["split"] == "val"].copy()

    # Weighted sampling to balance classes per batch
    labels = train_df["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels)
    class_counts = np.maximum(class_counts, 1)  # safety
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    augment = bool(cfg.get("train", {}).get("augment", False))

    train_ds = CbisDicomDataset(train_df, img_size=img_size, augment=augment)
    val_ds = CbisDicomDataset(val_df, img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = get_device()
    model = make_resnet18_binary(pretrained=True).to(device)

    ####### Experiment 1: optionally freeze backbone (feature extraction) #######
    if cfg.get("train", {}).get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    ##############################################################################

    criterion = make_bce_loss(cfg, device)

    base_lr = float(cfg["train"]["lr"])
    optimiser = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": base_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": base_lr * 0.1},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    # Training loop
    epochs = int(cfg["train"]["epochs"])
    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.float32)

            optimiser.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimiser.step()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

            pbar.set_postfix(train_loss=(running_loss / max(seen, 1)))

        # Validation at end of epoch
        val_metrics = evaluate(model, val_loader, device=device, cfg=cfg)
        scheduler.step(val_metrics["val_loss"])
        train_loss = running_loss / max(seen, 1)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_acc={val_metrics['val_acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]

            ckpt_path = run_dir / "model_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint (AUC={best_val_auc:.4f}): {ckpt_path}")

    # Save final model
    final_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    torch.save(model.state_dict(), processed_dir / "model_final.pt")


if __name__ == "__main__":
    main()
