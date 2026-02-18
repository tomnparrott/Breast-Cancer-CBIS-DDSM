from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pydicom


def load_first_dicom_as_array(series_dir: Path) -> np.ndarray:
    dcm_files = sorted(series_dir.rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    ds = pydicom.dcmread(str(dcm_files[0]))

    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Apply windowing if present (handles MultiValue)
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        # window center/width can be pydicom.multival.MultiValue
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

    return img


class CbisDicomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, augment: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.augment = augment

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
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        patient_id = str(row["patient_id"])
        label_int = int(row["label"])
        series_dir = Path(row["image_dir"])

        img = load_first_dicom_as_array(series_dir)

        x = self.to_tensor(img)  
        x = self.resize(x)
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
