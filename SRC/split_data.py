from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

def split_by_patient(
    manifest: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    random_state: int
) -> pd.DataFrame:

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=random_state
    )

    train_idx, test_idx = next(
        gss_test.split(manifest, groups=manifest["patient_id"])
    )

    train_df = manifest.iloc[train_idx].copy()
    test_df = manifest.iloc[test_idx].copy()

    adjusted_val_frac = val_frac / (1.0 - test_frac)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=adjusted_val_frac,
        random_state=random_state
    )

    train_idx, val_idx = next(
        gss_val.split(train_df, groups=train_df["patient_id"])
    )

    final_train = train_df.iloc[train_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    final_train["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([final_train, val_df, test_df], ignore_index=True)

    return split_df


def main() -> None:
    cfg_path = Path("Configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    seed = int(cfg.get("seed", 42))
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    val_frac = float(cfg["split"]["val_frac"])
    test_frac = float(cfg["split"]["test_frac"])

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

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

    (split_df[split_df["split"] == "train"]).to_csv(processed_dir / "train.csv", index=False)
    (split_df[split_df["split"] == "val"]).to_csv(processed_dir / "val.csv", index=False)
    (split_df[split_df["split"] == "test"]).to_csv(processed_dir / "test.csv", index=False)

    # Print summary
    print("\n=== SPLIT SUMMARY ===")
    print(f"Total rows: {len(split_df)}")
    print(f"Train rows: {int((split_df['split'] == 'train').sum())}")
    print(f"Val rows:   {int((split_df['split'] == 'val').sum())}")
    print(f"Test rows:  {int((split_df['split'] == 'test').sum())}")
    print(f"Saved: {splits_path}\n")


if __name__ == "__main__":
    main()
