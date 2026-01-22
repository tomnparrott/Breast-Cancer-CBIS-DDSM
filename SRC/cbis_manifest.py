from pathlib import Path
import pandas as pd
from typing import Optional


# Convert pathology text labels into binary class labels
def pathology_to_label(pathology: str) -> int:
    # Standardise pathology labels to binary values: malignant=1, benign=0
    pathology = str(pathology).lower()
    if "malignant" in pathology:
        return 1
    return 0


def _find_csv_upwards(start_dir: Path, filename: str, max_levels: int = 4) -> Optional[Path]:
    d = start_dir
    for _ in range(max_levels + 1):
        candidate = d / filename
        if candidate.exists():
            return candidate
        d = d.parent
    return None


# Build a unified data table, linking image paths to labels & patient IDs
def build_tcia_manifest(raw_dir: Path) -> pd.DataFrame:
    csv_files = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]


    frames = []
    for name in csv_files:
        csv_path = _find_csv_upwards(raw_dir, name, max_levels=4)
        if csv_path is not None:
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise RuntimeError(
            "No CBIS-DDSM CSV files found. "
            "Expected files like mass_case_description_train_set.csv. "
            "Place them in the raw_dir or a parent folder."
        )

    # Combine all metadata tables into one df
    df = pd.concat(frames, ignore_index=True)

    # Standardise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns for downstream processing
    required_columns = {"patient_id", "pathology", "image file path"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise RuntimeError(f"Missing required columns: {missing}")

    # Root directory containing all CBIS-DDSM image folders
    cbis_root = raw_dir / "CBIS-DDSM"
    if not cbis_root.exists():
        raise RuntimeError(f"CBIS-DDSM folder not found at: {cbis_root}")

    records = []

    # Iterate through the metadata rows and resolve image dirs
    for _, row in df.iterrows():
        image_path = Path(str(row["image file path"]))

        parts = image_path.parts

        # Find the real case folder anywhere in the path (e.g., Calc-... or Mass-...)
        folder_name = None
        for p in parts:
            if p.startswith("Calc-") or p.startswith("Mass-"):
                folder_name = p
                break

        if folder_name is None:
            continue

        image_dir = cbis_root / folder_name
        if not image_dir.exists():
            continue

        records.append(
            {
                "patient_id": str(row["patient_id"]),
                "label": pathology_to_label(row["pathology"]),
                "image_dir": str(image_dir),
            }
        )

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError("Manifest is empty — image paths not resolving")

    return manifest
