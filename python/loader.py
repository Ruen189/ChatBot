"""Модуль для загрузки конфигурации и проверки API-ключей."""
from typing import Optional, Dict
from vllm import SamplingParams
import yaml
from fastapi import HTTPException, Header
from pathlib import Path
from python.textblock_formatter import build_block
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

def load_yaml(path: str = "config.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_txt(path: str) -> str:
    """Загрузка текстового файла."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
CONFIG_PATH = Path("config.yaml")
config = load_yaml(CONFIG_PATH)
API_KEYS = set(config.get("api_keys", []))

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Проверка API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    