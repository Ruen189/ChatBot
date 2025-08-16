import re
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from .loader import config, load_yaml, load_txt
from .textblock_formatter import build_courses_block, build_locations_block
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

llm: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None
system_prompt: str = ""

CONFIG_PATH = Path(config["paths"]["samples"])
SYSTEM_PROMPT_PATH = Path(config["paths"]["system_prompt"])
COURSES_PATH = Path(config["paths"]["courses"])
LOCATIONS_PATH = Path(config["paths"]["locations"])



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

class ConfigWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH.name):
            load_sampling_params()
        elif event.src_path.endswith(SYSTEM_PROMPT_PATH.name) \
                or event.src_path.endswith(LOCATIONS_PATH.name) \
                or event.src_path.endswith(COURSES_PATH.name):
            load_system_prompt()
            
@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения."""
    global llm
    engine_args = AsyncEngineArgs(
        model=config["model"].get("name", "PrunaAI/IlyaGusev-saiga_mistral_7b_merged-AWQ-4bit-smashed"),
        quantization=config["model"].get("quantization", "awq_marlin"),
        gpu_memory_utilization=config["model"].get("gpu_memory_utilization", 0.6),
        dtype=config["model"].get("dtype", "auto"),
        max_model_len=config["model"].get("max_model_len", None),
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    
    load_sampling_params()
    load_system_prompt()
    
    # Мини-тест на прогрев
    async for _ in llm.generate(
        request_id="warmup",
        prompt=system_prompt,
        sampling_params=SamplingParams(max_tokens=5)
    ):
        break

    event_handler = ConfigWatcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()

    yield
    observer.stop()
    observer.join()


async def get_llm_reply(context: list) -> AsyncGenerator[str, None]:
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

    async for output in llm.generate(
        request_id="chat-stream",
        prompt=prompt,
        sampling_params=sampling_params
    ):
        for completion in output.outputs:
            if completion.text:
                yield completion.text