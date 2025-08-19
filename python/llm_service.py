"""Модуль для работы с LLM-сервисом: инициализация, генерация ответов и мониторинг конфигов."""
import uuid
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
import python.loader as loader
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

llm: Optional[AsyncLLM] = None
system_prompt: str = ""

def load_LLM():
    """Загружает модель LLM на основе конфигурации."""
    global llm
    if loader.MODEL_PATH is None:
        raise RuntimeError("MODEL_PATH не установлен. Проверь config.yaml -> paths.model")

    if isinstance(llm, AsyncLLM):
        del llm
        
    cfg = loader.load_yaml(loader.MODEL_PATH)
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
    """Собирает все блоки данных в один system_prompt."""
    global system_prompt
    data_blocks = loader.load_data_blocks()
    system_prompt = "\n\n".join(data_blocks.values())
    print(f"System prompt сформирован, длина: {len(system_prompt)} символов")
    return system_prompt


class ConfigWatcher(FileSystemEventHandler):
    """Автоматическая перезагрузка конфигурации при изменении файлов."""
    def on_modified(self, event):
        try:
            path = Path(event.src_path)
            if path.name == loader.CONFIG_PATH.name:
                loader.load_paths()
                loader.load_sampling_params()
                load_LLM()
                build_system_prompt()
            elif loader.SAMPLES_PATH and path.name == loader.SAMPLES_PATH.name:
                loader.load_sampling_params()
            elif loader.MODEL_PATH and path.name == loader.MODEL_PATH.name:
                load_LLM()
            elif path in loader.DATA_PATHS.values():
                build_system_prompt()
        except Exception as e:
            print("Ошибка в ConfigWatcher:")
            traceback.print_exc()

def generate_answer(prompt: str, sampling_params: SamplingParams, request_id: Optional[str] = None):
    if llm is None:
        raise RuntimeError("LLM не загружена. Проверь конфиг.")
    if request_id is None:
        request_id = f"chat-{uuid.uuid4().hex}"
    return llm.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params
    )

async def warmup_llm():
    """Функция для прогрева LLM, чтобы избежать задержек при первом запросе."""
    if llm is None:
        raise RuntimeError("LLM не загружена. Проверь конфиг.")
    try:
        async for _ in generate_answer(system_prompt + "Инициализация модели", SamplingParams(max_tokens=50)):
            pass
    except Exception as e:
        print(f"Ошибка при прогреве LLM: {e}")
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения."""
    try:
        loader.load_paths()
        loader.load_sampling_params()
        build_system_prompt()
        load_LLM()

        await warmup_llm()
        print("LLM инициализирован")
    except Exception as e:
        print("Ошибка при инициализации LLM:")
        traceback.print_exc()
        
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
    async for output in generate_answer(prompt, loader.sampling_params, request_id):
        for completion in output.outputs:
            if completion.text:
                yield completion.text
