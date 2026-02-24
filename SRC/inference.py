from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import argparse
import json
import time
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from SRC.model import make_resnet18_binary
from SRC.dataset import load_dicom_as_array, PadToSquare


# -----------------------------
# Paths / Run discovery
# -----------------------------
def project_root() -> Path:
    # SRC/ -> project root is one level up
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = project_root() / "Configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at: {cfg_path}")
    import yaml
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def processed_dir_from_cfg(cfg: dict) -> Path:
    return project_root() / Path(cfg["data"]["processed_dir"])


def resolve_latest_run_dir(processed_dir: Path) -> Path:
    """
    Mirrors eval.py behaviour:
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


def list_run_dirs(processed_dir: Path) -> list[Path]:
    runs_root = processed_dir / "runs"
    if not runs_root.exists():
        return []
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def resolve_checkpoint_path(run_dir: Path, ckpt: str) -> Path:
    ckpt = ckpt.strip().lower()
    if ckpt in {"best", "model_best", "model_best.pt"}:
        p = run_dir / "model_best.pt"
    elif ckpt in {"final", "model_final", "model_final.pt"}:
        p = run_dir / "model_final.pt"
    else:
        p = run_dir / ckpt

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


# -----------------------------
# Preprocessing
# -----------------------------
@dataclass
class PreprocessOutput:
    x: torch.Tensor          # (1, C, H, W) model-ready tensor
    img_display: np.ndarray  # (H, W) float in [0,1] (post crop/pad/resize)


def preprocess_series_dir(
    series_dir: Path,
    img_size: int,
    num_channels: int = 3,
    crop_foreground: bool = True,
) -> PreprocessOutput:
    """
    Loads a DICOM series folder and applies:
      load_dicom_as_array -> ToTensor -> PadToSquare -> Resize -> repeat channels -> Normalize
    """
    if not series_dir.exists():
        raise FileNotFoundError(f"Series dir does not exist: {series_dir}")

    img = load_dicom_as_array(series_dir, crop_foreground=crop_foreground).astype(np.float32)  # (H,W), [0,1]

    to_tensor = transforms.ToTensor()  # (1,H,W)
    pad = PadToSquare()
    resize = transforms.Resize((img_size, img_size))

    x = to_tensor(img)
    x = pad(x)
    x = resize(x)

    img_disp = x.squeeze(0).numpy().astype(np.float32)

    if num_channels == 3:
        x = x.repeat(3, 1, 1)
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        norm = transforms.Normalize(mean=[0.5], std=[0.25])

    x = norm(x)
    x = x.unsqueeze(0)  # (1,C,H,W)
    return PreprocessOutput(x=x, img_display=img_disp)


# -----------------------------
# Model loading
# -----------------------------
def load_model_from_checkpoint(ckpt_path: Path, device: str) -> nn.Module:
    """
    Loads ResNet18 binary model and applies checkpoint weights.
    Uses pretrained=False because weights are loaded from your checkpoint.
    """
    model = make_resnet18_binary(pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------
# Prediction (with optional TTA)
# -----------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, device: str, tta: bool = False) -> float:
    """
    Returns P(y=1). Optional simple TTA: base + horizontal flip average.
    """
    x = x.to(device=device, dtype=torch.float32)

    logits = model(x).view(-1)
    p = torch.sigmoid(logits)[0]

    if tta:
        x_flip = torch.flip(x, dims=[3])  # flip width
        logits_f = model(x_flip).view(-1)
        p_f = torch.sigmoid(logits_f)[0]
        p = 0.5 * (p + p_f)

    return float(p.item())


def decision_from_threshold(prob: float, threshold: float) -> int:
    return int(prob >= threshold)


# -----------------------------
# Grad-CAM (target: layer4[-1].conv2)
# -----------------------------
def _get_default_cam_layer(model: nn.Module) -> nn.Module:
    """
    For torchvision ResNet18, layer4 is a Sequential of BasicBlocks.
    The most commonly used Grad-CAM target is the last conv in the last block.
    """
    try:
        return model.layer4[-1].conv2
    except Exception as e:
        raise RuntimeError("Could not resolve Grad-CAM layer as model.layer4[-1].conv2") from e


def gradcam_resnet18(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    """
    Grad-CAM on a target conv layer (default: model.layer4[-1].conv2).
    Returns heatmap in [0,1] shaped (H,W) aligned to input.
    """
    model.eval()
    x = x.to(device=device, dtype=torch.float32)

    layer = target_layer if target_layer is not None else _get_default_cam_layer(model)

    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_m, _gin, gout):
        nonlocal gradients
        gradients = gout[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x).view(-1)
    score = logits[0]
    score.backward()

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks failed (no activations/gradients).")

    # activations/gradients: (1,C,h,w)
    weights = gradients.mean(dim=(2, 3), keepdim=True)       # (1,C,1,1)
    cam = (weights * activations).sum(dim=1).squeeze(0)      # (h,w)
    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy().astype(np.float32)

    return cam


# -----------------------------
# Unseen data helpers (zip -> temp dir)
# -----------------------------
@dataclass
class ExtractedZip:
    series_dir: Path
    _tmp: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        self._tmp.cleanup()


def extract_zip_to_temp(zip_bytes: bytes) -> ExtractedZip:
    """
    Extract uploaded ZIP of DICOMs to a temporary directory.
    Return an object you must keep alive while using the extracted path.
    """
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    zpath = root / "upload.zip"
    zpath.write_bytes(zip_bytes)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(root / "unzipped")
    return ExtractedZip(series_dir=root / "unzipped", _tmp=tmp)


# -----------------------------
# High-level API: run inference
# -----------------------------
def run_inference(
    series_dir: Path,
    ckpt: str = "best",
    threshold: float = 0.5,
    tta: bool = False,
    explain: bool = True,
    prefer_cuda: bool = True,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    One-call inference:
      - loads config
      - resolves latest run dir (unless run_dir provided)
      - loads checkpoint
      - preprocesses series_dir
      - predicts probability (+ optional TTA)
      - optional Grad-CAM (layer4[-1].conv2)
      - returns structured result for Streamlit/logging
    """
    cfg = load_config()
    proc_dir = processed_dir_from_cfg(cfg)

    resolved_run_dir = run_dir if run_dir is not None else resolve_latest_run_dir(proc_dir)
    ckpt_path = resolve_checkpoint_path(resolved_run_dir, ckpt)

    device = get_device(prefer_cuda=prefer_cuda)
    model = load_model_from_checkpoint(ckpt_path, device=device)

    img_size = int(cfg["data"]["img_size"])
    prep = preprocess_series_dir(series_dir, img_size=img_size, num_channels=3, crop_foreground=True)

    t0 = time.time()
    prob = predict_proba(model, prep.x, device=device, tta=tta)
    pred = decision_from_threshold(prob, threshold)
    dt = time.time() - t0

    result: Dict[str, Any] = {
        "series_dir": str(series_dir),
        "run_dir": str(resolved_run_dir),
        "checkpoint": str(ckpt_path),
        "device": device,
        "img_size": img_size,
        "tta": bool(tta),
        "threshold": float(threshold),
        "prob_malignant": float(prob),
        "pred_label": int(pred),
        "latency_s": float(dt),
        "img_display": prep.img_display,  # (H,W) float32
    }

    if explain:
        cam = gradcam_resnet18(model, prep.x, device=device, target_layer=None)
        result["gradcam"] = cam  # (H,W) float32

    return result


# -----------------------------
# CLI (quick verification)
# -----------------------------
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Run inference on a DICOM series folder.")
    ap.add_argument("--series_dir", type=str, required=True, help="Path to a folder containing DICOMs.")
    ap.add_argument("--ckpt", type=str, default="best", help="best|final|<filename>")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no_explain", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--run_dir", type=str, default="", help="Optional explicit run dir (overrides latest_run.txt).")
    ap.add_argument("--out_json", type=str, default="", help="Optional path to save JSON output.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir.strip() else None

    res = run_inference(
        series_dir=Path(args.series_dir),
        ckpt=args.ckpt,
        threshold=args.threshold,
        tta=bool(args.tta),
        explain=not bool(args.no_explain),
        prefer_cuda=not bool(args.cpu),
        run_dir=run_dir,
    )

    # make JSON serialisable (exclude arrays)
    serialisable = {k: v for k, v in res.items() if k not in {"img_display", "gradcam"}}
    print(json.dumps(serialisable, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    _cli()