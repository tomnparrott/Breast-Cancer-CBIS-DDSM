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

# Resolve the repository root
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

# Load the shared config so inference uses the same settings as training
def load_config() -> dict:
    cfg_path = project_root() / "Configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at: {cfg_path}")
    import yaml
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# Choose the best available inference device
def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Resolve the processed data directory from config
def processed_dir_from_cfg(cfg: dict) -> Path:
    return project_root() / Path(cfg["data"]["processed_dir"])

# Resolve the latest run directory from the processed data folder, which is where checkpoints are loaded from by default
def resolve_latest_run_dir(processed_dir: Path) -> Path:
    latest_ptr = processed_dir / "latest_run.txt"
    if latest_ptr.exists():
        run_id = latest_ptr.read_text(encoding="utf-8").strip()
        candidate = processed_dir / "runs" / run_id
        if candidate.exists():
            return candidate
    return processed_dir

# List all run directories available in the processed data folder
def list_run_dirs(processed_dir: Path) -> list[Path]:
    runs_root = processed_dir / "runs"
    if not runs_root.exists():
        return []
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs

# Map checkpoint names to actual files
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

# Preprocessing #
# Bundle the model input tensor together with the display image
@dataclass
class PreprocessOutput:
    x: torch.Tensor
    img_display: np.ndarray

# Load a DICOM series folder and apply the same preprocessing used during training, returning both the model input tensor and the display image for Grad-CAM
def preprocess_series_dir(
    series_dir: Path,
    img_size: int,
    num_channels: int = 3,
    crop_foreground: bool = True,
) -> PreprocessOutput:
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

# Rebuild the model architecture and load checkpoint weights onto it
def load_model_from_checkpoint(ckpt_path: Path, device: str) -> nn.Module:
    model = make_resnet18_binary(pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# Run the model and return a malignant probability
@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, device: str, tta: bool = False) -> float:
    x = x.to(device=device, dtype=torch.float32)
    # Run the model forward and apply sigmoid to get a probability
    logits = model(x).view(-1)
    p = torch.sigmoid(logits)[0]

    if tta:
        x_flip = torch.flip(x, dims=[3])  # flip width
        logits_f = model(x_flip).view(-1)
        p_f = torch.sigmoid(logits_f)[0]
        p = 0.5 * (p + p_f)

    return float(p.item())

# Convert a probability into the binary class label
def decision_from_threshold(prob: float, threshold: float) -> int:
    return int(prob >= threshold)

# Grad-CAM 
def _get_default_cam_layer(model: nn.Module) -> nn.Module:
    try:
        return model.layer4[-1].conv2
    except Exception as e:
        raise RuntimeError("Could not resolve Grad-CAM layer as model.layer4[-1].conv2") from e

# Compute Grad-CAM heatmap for a given input and model, returning a (H,W) array in [0,1] aligned to the input image. Uses the last conv layer of ResNet18
def gradcam_resnet18(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    model.eval()
    x = x.to(device=device, dtype=torch.float32)
    # If no target layer specified, default to the last conv layer of ResNet18
    layer = target_layer if target_layer is not None else _get_default_cam_layer(model)

    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    # Define hooks to capture the activations and gradients from the target layer during the forward and backward passes, which are needed to compute the Grad-CAM heatmap. The forward hook saves the output of the target layer, and the backward hook saves the gradients flowing back from the output with respect to the target layer's output
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
        raise RuntimeError("Grad-CAM hooks failed")

    # activations/gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze(0)
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

# Keep the temporary extraction directory while inference is using it
@dataclass
class ExtractedZip:
    series_dir: Path
    _tmp: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        self._tmp.cleanup()

# Extract an uploaded ZIP of DICOM files into a temporary folder
def extract_zip_to_temp(zip_bytes: bytes) -> ExtractedZip:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    zpath = root / "upload.zip"
    zpath.write_bytes(zip_bytes)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(root / "unzipped")
    return ExtractedZip(series_dir=root / "unzipped", _tmp=tmp)

# Tie together config loading, preprocessing, prediction, and Grad-CAM
def run_inference(
    series_dir: Path,
    ckpt: str = "best",
    threshold: float = 0.5,
    tta: bool = False,
    explain: bool = True,
    prefer_cuda: bool = True,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Load config and resolve paths, then load the model checkpoint
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

    # Bundle results together into a dictionary
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

# Provide a command-line entry point for quick local inference checks
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

    # Run inference and print the results as JSON
    res = run_inference(
        series_dir=Path(args.series_dir),
        ckpt=args.ckpt,
        threshold=args.threshold,
        tta=bool(args.tta),
        explain=not bool(args.no_explain),
        prefer_cuda=not bool(args.cpu),
        run_dir=run_dir,
    )

    # make JSON serialisable 
    serialisable = {k: v for k, v in res.items() if k not in {"img_display", "gradcam"}}
    print(json.dumps(serialisable, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    _cli()