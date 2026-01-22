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

    # Normalise to [0, 1]
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
            ]
        )
        self.to_tensor = transforms.ToTensor()  

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

        if self.augment:
            x = self.aug(x)

        y = torch.tensor([label_int], dtype=torch.float32)

        return {
            "image": x,
            "label": y,
            "patient_id": patient_id,
        }
