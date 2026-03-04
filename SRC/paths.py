from pathlib import Path
import yaml


# Resolve the repository root from the location of this file
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Load the shared config file used by the pipeline
def load_config() -> dict:
    cfg_path = project_root() / "Configs" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


# Return the raw and processed data directories defined in config
def get_data_dirs(cfg: dict) -> tuple[Path, Path]:
    root = project_root()
    raw_dir = root / cfg["data"]["raw_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    return raw_dir, processed_dir
