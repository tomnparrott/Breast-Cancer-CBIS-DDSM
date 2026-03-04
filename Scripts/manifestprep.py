from pathlib import Path
import yaml
import pandas as pd

from SRC.cbis_manifest import build_tcia_manifest


# Build the processed manifest CSV from the raw CBIS-DDSM metadata and folders
def main() -> None:
    # Load the config to find the raw and processed data directories
    cfg_path = Path("Configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    # Ensure the processed output folder exists before writing the manifest
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create the unified manifest table used by the rest of the pipeline
    manifest = build_tcia_manifest(raw_dir)

    # Print a quick summary so the manifest can be checked at a glance
    print("\n=== MANIFEST SUMMARY ===")
    print(f"Rows: {len(manifest)}")
    print(f"Unique patients: {manifest['patient_id'].nunique()}")
    print("Label counts (0=benign, 1=malignant):")
    print(manifest["label"].value_counts(dropna=False).sort_index())

    missing_dirs = (manifest["image_dir"].astype(str).str.len() == 0).sum()
    print(f"Empty image_dir rows: {missing_dirs}")

    # Save the manifest for downstream splitting, training, and evaluation
    out_path = processed_dir / "manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")


if __name__ == "__main__":
    main()
