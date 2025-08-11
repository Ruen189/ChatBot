import logging
from pathlib import Path
import yaml

def load_yaml_data(path: Path) -> dict:
    """Загрузка и проверка данных из YAML файла."""
    if not path.exists():
        logging.error(f"YAML data file not found: {path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logging.error(f"YAML data in {path} is not a dict")
            return {}
        return data
    except yaml.YAMLError as e:
        logging.error(f"YAML decode error in file {path}: {e}")
        return {}
