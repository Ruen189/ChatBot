"""Модуль для работы с LLM-сервисом, включая инициализацию, загрузку конфигурации и генерацию ответов."""
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from python.loader import config, load_yaml, load_txt, CONFIG_PATH
from python.textblock_formatter import build_block
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid

llm: Optional[AsyncLLM] = None
sampling_params: Optional[SamplingParams] = None
system_prompt: str = ""

SAMPLES_PATH = None
MODEL_PATH = None
DATA_PATHS: Dict[str, Path] = {}


def load_paths():
    """Загружает пути из конфигурации."""
    global SAMPLES_PATH, MODEL_PATH, DATA_PATHS
    # Берем пути из config.yaml
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
    """
    Загружает все файлы из config["paths"] (кроме samples и model).
    Поддерживаются yaml и txt.
    """
    blocks = {}

    for name, path in DATA_PATHS.items():
        try:
            if path.suffix == ".txt":
                # Обычный текстовый блок
                blocks[name] = load_txt(path)

            elif path.suffix in (".yaml", ".yml"):
                data = load_yaml(path)
                if isinstance(data, dict) and len(data) == 1:
                    key = list(data.keys())[0]
                    value = data[key]
                else:
                    key = name
                    value = data
                blocks[name] = build_block(value, key)

            else:
                print(f"Формат файла {path} не поддерживается")

        except Exception as e:
            print(f"Ошибка загрузки {name} из {path}: {e}")

    return blocks

def load_LLM():
    global llm
    if isinstance(llm, AsyncLLM):
        del llm
    cfg = load_yaml(MODEL_PATH)
    engine_args = AsyncEngineArgs(
        model=cfg["model"].get("name", "PrunaAI/IlyaGusev-saiga_mistral_7b_merged-AWQ-4bit-smashed"),
        quantization=cfg["model"].get("quantization", "awq_marlin"),
        gpu_memory_utilization=cfg["model"].get("gpu_memory_utilization", 0.6),
        dtype=cfg["model"].get("dtype", "auto"),
        max_model_len=cfg["model"].get("max_model_len", None),
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    
    print(f"Загружена модель")

def build_system_prompt() -> str:
    """
    Склеивает все блоки данных в один system_prompt.
    """
    global system_prompt
    data_blocks = load_data_blocks()
    system_prompt = "\n\n".join(data_blocks.values())
    print(f"System prompt: {system_prompt}")
    return system_prompt
    
class ConfigWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        path = Path(event.src_path)
        if path.name == CONFIG_PATH.name:
            load_paths()
            load_sampling_params()
            load_LLM()
            build_system_prompt()
        elif path.name == SAMPLES_PATH.name:
            load_sampling_params()
        elif path.name == MODEL_PATH.name:
            load_LLM()
        elif path in DATA_PATHS.values():
            build_system_prompt()

            
def generate_answer(prompt: str, sampling_params: SamplingParams, request_id: Optional[str] = None):
    if request_id is None:
        request_id = f"chat-{uuid.uuid4().hex}"
    return llm.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params
    )   


@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения."""
    load_paths()
    load_sampling_params()
    build_system_prompt()
    load_LLM()
    
    async for _ in generate_answer("Инициализация модели", SamplingParams(max_tokens=50)):
        break
    print("LLM инициализирован")
    
    event_handler = ConfigWatcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()

    yield
    
    observer.stop()
    observer.join()

async def get_llm_reply(context: list, request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Генерация ответа LLM на основе контекста."""
    prompt_parts = [system_prompt, "Диалог с пользователем:"]
    for msg in context:
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "user":
            prompt_parts.append(f"Пользователь: {content}")
        elif role == "bot":
            prompt_parts.append(f"Бот: {content}")
    prompt_parts.append("Бот:")

    prompt = "\n".join(prompt_parts)

    async for output in generate_answer(prompt, sampling_params, request_id):
        for completion in output.outputs:
            if completion.text:
                yield completion.text