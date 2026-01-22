from pathlib import Path
import yaml
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += float(loss.item()) * x.size(0)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def main() -> None:
    cfg = load_config()
    seed = int(cfg["seed"])
    set_seed(seed)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

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
    splits_path = processed_dir / "splits.csv"
    splits.to_csv(splits_path, index=False)

    img_size = int(cfg["data"]["img_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_df = splits[splits["split"] == "train"].copy()
    val_df = splits[splits["split"] == "val"].copy()

    train_ds = CbisDicomDataset(train_df, img_size=img_size, augment=True)
    val_ds = CbisDicomDataset(val_df, img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    criterion = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # Training loop
    epochs = int(cfg["train"]["epochs"])
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimiser.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimiser.step()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

            pbar.set_postfix(train_loss=(running_loss / max(seen, 1)))

        # Validation at end of epoch
        val_metrics = evaluate(model, val_loader, device=device)
        train_loss = running_loss / max(seen, 1)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_acc={val_metrics['val_acc']:.4f}"
        )

        # Save best model checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

            ckpt_path = processed_dir / "model_best.pt"
            torch.save(model.state_dict(), ckpt_path)

            print(f"Saved best checkpoint: {ckpt_path}")

    # Save final model
    final_path = processed_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
