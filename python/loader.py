import yaml
from typing import Optional
from fastapi import HTTPException, Header

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

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Проверка API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
