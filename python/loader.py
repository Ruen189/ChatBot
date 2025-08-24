"""Модуль для загрузки конфигурации, текстовых блоков и проверки API-ключей."""
from typing import Optional, Dict
import yaml
from fastapi import HTTPException, Header
from pathlib import Path
from python.textblock_formatter import build_block
from vllm import SamplingParams

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

SAMPLES_PATH: Optional[Path] = None
MODEL_PATH: Optional[Path] = None
DATA_PATHS: Dict[str, Path] = {}
sampling_params: Optional[SamplingParams] = None


def load_paths():
    """Загружает пути из конфигурации."""
    global SAMPLES_PATH, MODEL_PATH, DATA_PATHS
    if "paths" not in config:
        raise ValueError("Отсутствует секция 'paths' в конфигурации")
    if "samples" not in config["paths"] or "model" not in config["paths"]:
        raise ValueError("Отсутствуют пути 'samples' или 'model' в секции 'paths' конфигурации")        

    SAMPLES_PATH = Path(config["paths"]["samples"])
    MODEL_PATH = Path(config["paths"]["model"])
    DATA_PATHS = {
        k: Path(v) for k, v in config["paths"].items()
        if k not in ["samples", "model"]
    }


def load_sampling_params():
    """Загружает параметры семплинга из файла samples."""
    global sampling_params
    cfg = load_yaml(SAMPLES_PATH)
    sampling_params = SamplingParams(
        temperature=cfg["sampling"].get("temperature", 0.3),
        top_p=cfg["sampling"].get("top_p", 0.5),
        max_tokens=cfg["sampling"].get("max_tokens", 350),
        stop=cfg["sampling"].get("stop", "Пользователь:"),
    )
    print(f"Загружены параметры сэмплинга")

def load_data_blocks() -> Dict[str, str]:
    blocks = {}
    for name, path in DATA_PATHS.items():
        try:
            if path.suffix == ".txt":
                blocks[name] = load_txt(path)

            elif path.suffix in (".yaml", ".yml"):
                raw = load_txt(path)
                data = yaml.safe_load(raw)

                if isinstance(data, dict) and len(data.keys()) == len(set(data.keys())):
                    for k, v in data.items():
                        blocks[f"{name}_{k}"] = build_block(v, k)

                elif isinstance(data, dict):
                    root = yaml.compose(raw)
                    items = []
                    for node in root.value:
                        k = node[0].value
                        v = yaml.constructor.SafeConstructor.construct_object(yaml.SafeLoader, node[1])
                        items.append((k, v))

                    for idx, (k, v) in enumerate(items, start=1):
                        blocks[f"{name}_{k}_{idx}"] = build_block(v, k)

                elif isinstance(data, list):
                    for idx, item in enumerate(data, start=1):
                        key = f"{name}_dialog_{idx}"
                        blocks[key] = build_block(item, f"dialog_{idx}")

                else:
                    blocks[name] = str(data)

            else:
                print(f"Формат файла {path} не поддерживается")

        except Exception as e:
            print(f"Ошибка загрузки {name} из {path}: {e}")
    return blocks

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Проверка API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
