from pathlib import Path
import yaml
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import json
import datetime
import platform
import subprocess
from contextlib import nullcontext

from torch.amp import GradScaler, autocast  # fixes torch.cuda.amp.GradScaler deprecation

from SRC.seed import set_seed
from SRC.model import make_resnet18_binary
from SRC.dataset import CbisDicomDataset
from SRC.split_data import split_by_patient


# Load the main training config from disk
def load_config() -> dict:
    cfg_path = Path("Configs/config.yaml")
    return yaml.safe_load(cfg_path.read_text())


# Choose the best available device for training and validation
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Capture the current git commit for run metadata
def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# Create a timestamped run directory and update the latest-run pointer
def make_run_dir(processed_dir: Path, seed: int) -> Path:
    runs_root = processed_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S") + f"_seed{seed}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (processed_dir / "latest_run.txt").write_text(run_id, encoding="utf-8")
    return run_dir


# Save the environment and config snapshot needed to reproduce the run
def write_run_info(run_dir: Path, cfg: dict) -> None:
    cfg_path = Path("Configs/config.yaml")
    cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""

    info = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": get_git_hash(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu/mps",
        "seed": int(cfg.get("seed", -1)),
        "config_path": str(cfg_path),
        "config_snapshot": cfg_text,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# Use the configured positive-class weight or infer one from the training labels
def _resolve_pos_weight(cfg: dict, labels_np: np.ndarray) -> float:
    """
    Use cfg['train']['pos_weight'] if set to a sensible float.
    If it's 'auto' or <= 0, compute neg/pos from the TRAIN split.
    """
    pw = cfg.get("train", {}).get("pos_weight", 1.0)

    if isinstance(pw, str) and pw.strip().lower() == "auto":
        pw = 0.0

    pw = float(pw)

    if pw > 0:
        return pw

    pos = float((labels_np == 1).sum())
    neg = float((labels_np == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


# Build the weighted BCE loss used for binary classification
def make_bce_loss(pos_weight_value: float, device: str) -> nn.Module:
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Run one validation pass and return the core metrics tracked during training
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module) -> dict:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_probs = []

    use_amp = (device == "cuda")
    ctx = autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32).view(-1)

        with ctx:
            logits = model(x).view(-1)
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


# Train the model, validate each epoch, and save the run artifacts
def main() -> None:
    cfg = load_config()
    seed = int(cfg["seed"])
    set_seed(seed)

    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(processed_dir, seed)
    write_run_info(run_dir, cfg)

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run Scripts/manifestprep.py after the download completes."
        )

    manifest = pd.read_csv(manifest_path)

    splits = split_by_patient(
        manifest=manifest,
        val_frac=float(cfg["split"]["val_frac"]),
        test_frac=float(cfg["split"]["test_frac"]),
        random_state=seed,
    )

    splits.to_csv(run_dir / "splits.csv", index=False)
    splits.to_csv(processed_dir / "splits.csv", index=False)

    img_size = int(cfg["data"]["img_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    epochs = int(cfg["train"]["epochs"])

    train_df = splits[splits["split"] == "train"].copy()
    val_df = splits[splits["split"] == "val"].copy()

    labels_np = train_df["label"].astype(int).to_numpy()
    pos_weight_value = _resolve_pos_weight(cfg, labels_np)

    print(f"Train class counts: neg={(labels_np==0).sum()} pos={(labels_np==1).sum()}")
    print(f"Using pos_weight={pos_weight_value:.4f}")

    augment = bool(cfg.get("train", {}).get("augment", False))
    train_ds = CbisDicomDataset(train_df, img_size=img_size, augment=augment)
    val_ds = CbisDicomDataset(val_df, img_size=img_size, augment=False)

    use_weighted_sampler = bool(cfg.get("train", {}).get("use_weighted_sampler", False))

    pin_memory = torch.cuda.is_available()

    if use_weighted_sampler:
        class_counts = np.bincount(labels_np)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = get_device()
    model = make_resnet18_binary(pretrained=True).to(device)

    if cfg.get("train", {}).get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    criterion = make_bce_loss(pos_weight_value, device)

    base_lr = float(cfg["train"]["lr"])
    optimiser = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": base_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": base_lr * 0.1},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=[base_lr, base_lr * 0.1],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    use_amp = (device == "cuda")
    scaler = GradScaler(device="cuda", enabled=use_amp)  # fixes deprecation warning
    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.float32).view(-1)

            optimiser.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = model(x).view(-1)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # stability
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # IMPORTANT ORDER (removes scheduler warning):
            # optimizer step -> scheduler step -> scaler update
            scaler.step(optimiser)       # performs optimiser.step() when enabled
            scheduler.step()            # OneCycleLR expects stepping after optimiser.step()
            scaler.update()             # update scaler after stepping

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

            pbar.set_postfix(train_loss=(running_loss / max(seen, 1)), lr=optimiser.param_groups[0]["lr"])

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion)

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

    final_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    torch.save(model.state_dict(), processed_dir / "model_final.pt")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
