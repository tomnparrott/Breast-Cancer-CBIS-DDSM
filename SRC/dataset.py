from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

import pydicom


def _choose_best_dicom(series_dir: Path) -> Path:
    """
    Some CBIS-DDSM folders can contain multiple DICOMs (e.g., different derived images).
    This picks a 'best' candidate by favouring:
      - largest pixel area (H*W)
      - higher standard deviation (non-binary / non-empty looking images)
    """
    dcm_files = sorted(series_dir.rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    best_path: Optional[Path] = None
    best_score: float = -1.0

    for p in dcm_files:
        try:
            ds = pydicom.dcmread(str(p))
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        area = float(arr.size)
        std = float(arr.std())
        score = area + (std * 1000.0)

        if score > best_score:
            best_score = score
            best_path = p

    if best_path is None:
        raise FileNotFoundError(f"No readable DICOMs found in: {series_dir}")

    return best_path


def _apply_rescale_window_and_scale(ds, img: np.ndarray) -> np.ndarray:
    # Apply rescale slope/intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Apply windowing if present (handles MultiValue)
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        if isinstance(wc, (list, tuple)):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, (list, tuple)):
            ww = float(ww[0])
        else:
            ww = float(ww)

        lo = wc - (ww / 2.0)
        hi = wc + (ww / 2.0)
        img = np.clip(img, lo, hi)
    else:
        # robust clip if no windowing metadata
        lo, hi = np.percentile(img, (1, 99))
        if hi > lo:
            img = np.clip(img, lo, hi)

    # Scale to [0, 1]
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return img.astype(np.float32)


def _crop_to_foreground(img: np.ndarray, thr: float = 0.02, margin_frac: float = 0.03) -> np.ndarray:
    """
    Crops to non-background region to reduce black padding dominance.
    Assumes img is already in [0, 1].

    - thr: background threshold
    - margin_frac: extra border margin around detected foreground
    """
    if img.ndim != 2:
        raise ValueError("Expected a 2D (H, W) image array")

    mask = img > thr
    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h, w = img.shape
    margin = int(max(h, w) * margin_frac)

    y0 = max(0, y0 - margin)
    y1 = min(h - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(w - 1, x1 + margin)

    # Avoid degenerate crops
    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return img

    return img[y0 : y1 + 1, x0 : x1 + 1]


def load_dicom_as_array(series_dir: Path, crop_foreground: bool = True) -> np.ndarray:
    dcm_path = _choose_best_dicom(series_dir)
    ds = pydicom.dcmread(str(dcm_path))

    img = ds.pixel_array.astype(np.float32)
    img = _apply_rescale_window_and_scale(ds, img)

    if crop_foreground:
        img = _crop_to_foreground(img, thr=0.02, margin_frac=0.03)

    return img


class PadToSquare:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (C, H, W)
        _, h, w = x.shape
        if h == w:
            return x

        # pad (left, top, right, bottom)
        if w > h:
            diff = w - h
            top = diff // 2
            bottom = diff - top
            padding = (0, top, 0, bottom)
        else:
            diff = h - w
            left = diff // 2
            right = diff - left
            padding = (left, 0, right, 0)

        return TF.pad(x, padding, fill=0)


class CbisDicomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        augment: bool,
        num_channels: int = 3,
        crop_foreground: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.num_channels = int(num_channels)
        self.crop_foreground = bool(crop_foreground)

        self.pad_to_square = PadToSquare()
        self.resize = transforms.Resize((img_size, img_size))

        self.aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            ]
        )

        self.to_tensor = transforms.ToTensor()

        # Default stays ImageNet norm for 3ch because you're using ImageNet-pretrained ResNet18
        if self.num_channels == 3:
            self.norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            # sensible default for single-channel if you ever switch
            self.norm = transforms.Normalize(mean=[0.5], std=[0.25])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        patient_id = str(row["patient_id"])
        label_int = int(row["label"])
        series_dir = Path(row["image_dir"])

        img = load_dicom_as_array(series_dir, crop_foreground=self.crop_foreground)

        x = self.to_tensor(img)          # (1, H, W), float32 already in [0,1]
        x = self.pad_to_square(x)
        x = self.resize(x)

        if self.num_channels == 3:
            x = x.repeat(3, 1, 1)

        if self.augment:
            x = self.aug(x)

        x = self.norm(x)
        y = torch.tensor([label_int], dtype=torch.float32)

        return {
            "image": x,
            "label": y,
            "patient_id": patient_id,
        }