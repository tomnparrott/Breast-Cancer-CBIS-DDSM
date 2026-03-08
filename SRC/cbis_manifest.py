from pathlib import Path
from typing import Optional, Any

import pandas as pd

# Convert pathology text labels into binary class labels
def pathology_to_label(pathology: str) -> int:
    pathology = str(pathology).lower()
    if "malignant" in pathology:
        return 1
    return 0

# Search parent folders CBIS metadata CSV files
def _find_csv_upwards(start_dir: Path, filename: str, max_levels: int = 4) -> Optional[Path]:
    d = start_dir
    for _ in range(max_levels + 1):
        candidate = d / filename
        if candidate.exists():
            return candidate
        d = d.parent
    return None

# Return the first populated value from a list of possible column names
def _first_present(row: pd.Series, keys: list[str]) -> Optional[Any]:
    for k in keys:
        if k in row.index:
            v = row[k]
            if pd.isna(v):
                continue
            return v
    return None

# Normalise density values into the string labels used later in the dashboard and subgroup analysis
def _parse_density(val: Any) -> str:
    if val is None or pd.isna(val):
        return "Unknown"
    try:
        # handles "3", 3.0, "3.0", anything that can be parsed as an int 1-4 is accepted, else "Unknown"
        i = int(float(str(val).strip()))
        if 1 <= i <= 4:
            return str(i)
        return "Unknown"
    except Exception:
        return "Unknown"

# Standardise laterality values for reporting and grouping
def _normalise_laterality(val: Any) -> str:
    if val is None or pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    if "LEFT" in s or s == "L":
        return "LEFT"
    if "RIGHT" in s or s == "R":
        return "RIGHT"
    return "Unknown"

# Standardise mammography view labels for reporting and grouping 
def _normalise_view(val: Any) -> str:
    if val is None or pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    if "CC" == s:
        return "CC"
    if "MLO" == s:
        return "MLO"
    if "CC" in s:
        return "CC"
    if "MLO" in s:
        return "MLO"
    return "Unknown"

# Gather the abnormality type from folder names or source CSV names
def _abnormality_from_folder(folder_name: Optional[str], source_csv: str) -> str:
    if folder_name:
        if folder_name.startswith("Calc-"):
            return "calcification"
        if folder_name.startswith("Mass-"):
            return "mass"

    s = str(source_csv).lower()
    if "calc" in s:
        return "calcification"
    if "mass" in s:
        return "mass"
    return "unknown"

# Build a unified data table, linking image paths to labels & patient IDs
def build_tcia_manifest(raw_dir: Path) -> pd.DataFrame:
    csv_files = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]
    # Search for the CSV files in the raw_dir and parent folders, and load them into dataframes
    frames: list[pd.DataFrame] = []
    for name in csv_files:
        csv_path = _find_csv_upwards(raw_dir, name, max_levels=4)
        if csv_path is not None:
            df_i = pd.read_csv(csv_path)
            df_i["_source_csv"] = name
            frames.append(df_i)
    # If no CSV files found, raise an error with instructions
    if not frames:
        raise RuntimeError(
            "No CBIS-DDSM CSV files found"
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

    # Candidate column names, some of them have different names, so this covers the common variations
    density_keys = ["breast density", "breast_density", "breast density (1-4)"]
    laterality_keys = ["left or right breast", "left_or_right_breast", "laterality"]
    view_keys = ["image view", "image_view", "view"]
    abnormality_id_keys = ["abnormality id", "abnormality_id", "abnormalityid"]

    records = []

    # Iterate through the metadata rows and resolve image dirs
    for _, row in df.iterrows():
        image_path = Path(str(row["image file path"]))
        parts = image_path.parts

        # Find the case folder anywhere in the path 
        folder_name = None
        for p in parts:
            if p.startswith("Calc-") or p.startswith("Mass-"):
                folder_name = p
                break
        if folder_name is None:
            continue

        # Resolve the image directory and check it exists
        image_dir = cbis_root / folder_name
        if not image_dir.exists():
            continue

        # Extract metadata for the manifest, using helper functions to handle variations and missing values
        source_csv = str(row.get("_source_csv", "unknown"))
        abnormality_type = _abnormality_from_folder(folder_name, source_csv)
        breast_density = _parse_density(_first_present(row, density_keys))
        laterality = _normalise_laterality(_first_present(row, laterality_keys))
        view = _normalise_view(_first_present(row, view_keys))

        abnormality_id = _first_present(row, abnormality_id_keys)
        abnormality_id = str(abnormality_id).strip() if abnormality_id is not None else "Unknown"

        records.append(
            {
                "patient_id": str(row["patient_id"]),
                "label": pathology_to_label(row["pathology"]),
                "image_dir": str(image_dir),
                "breast_density": breast_density,          
                "abnormality_type": abnormality_type,     
                "laterality": laterality,                  
                "view": view,                          
                "abnormality_id": abnormality_id,
                "source_csv": source_csv,
            }
        )

    # Convert the list of records into a df
    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError("Manifest is empty — image paths not resolving")

    return manifest