import yaml

def load_yaml(path: str = "config.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_txt(path: str) -> str:
    """Загрузка текстового файла."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


config = load_yaml()
API_KEYS = set(config.get("api_keys", []))
