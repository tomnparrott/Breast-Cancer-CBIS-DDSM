from pathlib import Path
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = project_root() / "Configs" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


def get_data_dirs(cfg: dict) -> tuple[Path, Path]:
    root = project_root()
    raw_dir = root / cfg["data"]["raw_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    return raw_dir, processed_dir
