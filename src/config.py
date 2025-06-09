import yaml
from pathlib import Path

def load_config(config_path: str = "../config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_gallery_path(config: dict) -> Path:
    return Path(config["gallery_path"])