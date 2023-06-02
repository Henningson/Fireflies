import yaml
from pathlib import Path

def read_config_yaml(file_path: str) -> dict:
    return yaml.safe_load(Path(file_path).read_text())

