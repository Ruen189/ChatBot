from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from python.loader import config, load_yaml, load_txt
from python.textblock_formatter import build_courses_block, build_locations_block
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid

llm: Optional[AsyncLLM] = None
sampling_params: Optional[SamplingParams] = None
system_prompt: str = ""

CONFIG_PATH = Path(config["paths"]["samples"])
SYSTEM_PROMPT_PATH = Path(config["paths"]["system_prompt"])
COURSES_PATH = Path(config["paths"]["courses"])
LOCATIONS_PATH = Path(config["paths"]["locations"])
MODEL_PATH = Path(config["paths"]["model"])


def load_sampling_params():
    global sampling_params
    cfg = load_yaml(CONFIG_PATH)
    sampling_params = SamplingParams(
        temperature=cfg["sampling"].get("temperature", 0.3),
        top_p=cfg["sampling"].get("top_p", 0.5),
        max_tokens=cfg["sampling"].get("max_tokens", 350),
        stop=cfg["sampling"].get("stop", "Пользователь:"),
    )
    print(f"Загружены параметры сэмплинга")
    
def load_system_prompt():
    global system_prompt
    system_txt = load_txt(SYSTEM_PROMPT_PATH)
    courses = load_yaml(COURSES_PATH)
    locations = load_yaml(LOCATIONS_PATH)
    system_prompt = (
        "Системный промпт:\n" +
        system_txt + "\n" +
        build_courses_block(courses) + "\n" +
        build_locations_block(locations) + "\n"
    )
    print(f"Загружен новый системный промпт")

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
    
class ConfigWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH.name):
            load_sampling_params()
        elif event.src_path.endswith(SYSTEM_PROMPT_PATH.name) \
                or event.src_path.endswith(LOCATIONS_PATH.name) \
                or event.src_path.endswith(COURSES_PATH.name):
            load_system_prompt()
        elif event.src_path.endswith(MODEL_PATH.name):
            load_LLM()

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
    load_sampling_params()
    load_system_prompt()
    load_LLM()
    
    async for _ in generate_answer(system_prompt+"Инициализация модели", SamplingParams(max_tokens=50)):
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