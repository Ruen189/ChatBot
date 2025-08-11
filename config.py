import yaml

def load_config(path: str = "config.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
API_KEYS = set(config.get("api_keys", []))
