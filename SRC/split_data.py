from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

# Split the manifest by patient so no patient appears in more than one split
def split_by_patient(
    manifest: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    random_state: int
) -> pd.DataFrame:
    # First split off the test set, then split the remaining train/val portion according to the adjusted val_frac
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=random_state
    )
    # Use patient_id as the grouping column to ensure no patient appears in more than one split to avoid data leakage
    train_idx, test_idx = next(
        gss_test.split(manifest, groups=manifest["patient_id"])
    )

    train_df = manifest.iloc[train_idx].copy()
    test_df = manifest.iloc[test_idx].copy()
    # Adjust the val_frac to account for the reduced size of the remaining train+val portion after splitting off the test set
    adjusted_val_frac = val_frac / (1.0 - test_frac)
    # split the remaining train_df into final train and val sets using the adjusted val_frac
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=adjusted_val_frac,
        random_state=random_state
    )

    train_idx, val_idx = next(
        gss_val.split(train_df, groups=train_df["patient_id"])
    )
    # Combine the splits back into one df
    final_train = train_df.iloc[train_idx].copy()
    val_df = train_df.iloc[val_idx].copy()
    # Add a column to indicate the split for each row, which is used by the dataset and training loop
    final_train["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([final_train, val_df, test_df], ignore_index=True)

    return split_df

# Load the manifest, create split files, and print a quick summary
def main() -> None:
    cfg_path = Path("Configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    # Set the random seed for reproducibility, and ensure the processed output folder exists
    seed = int(cfg.get("seed", 42))
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Get the split fractions from config, and check they are valid
    val_frac = float(cfg["split"]["val_frac"])
    test_frac = float(cfg["split"]["test_frac"])

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at: {manifest_path}")
    # Load the manifest and check required columns before splitting
    manifest = pd.read_csv(manifest_path)
    # Check required columns exist before splitting
    required_cols = {"patient_id", "label", "image_dir"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise RuntimeError(f"manifest.csv missing required columns: {missing}")

    split_df = split_by_patient(
        manifest=manifest,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=seed
    )

    # Save outputs
    splits_path = processed_dir / "splits.csv"
    split_df.to_csv(splits_path, index=False)

    # Also save separate CSVs for each split, which are used by the dataset and training loop
    (split_df[split_df["split"] == "train"]).to_csv(processed_dir / "train.csv", index=False)
    (split_df[split_df["split"] == "val"]).to_csv(processed_dir / "val.csv", index=False)
    (split_df[split_df["split"] == "test"]).to_csv(processed_dir / "test.csv", index=False)

    # Print summary
    print("\n Summary of splits:")
    print(f"Total rows: {len(split_df)}")
    print(f"Train rows: {int((split_df['split'] == 'train').sum())}")
    print(f"Val rows:   {int((split_df['split'] == 'val').sum())}")
    print(f"Test rows:  {int((split_df['split'] == 'test').sum())}")
    print(f"Saved: {splits_path}\n")

if __name__ == "__main__":
    main()