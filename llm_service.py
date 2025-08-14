import re
from contextlib import asynccontextmanager
from typing import Optional
from vllm import LLM, SamplingParams
from loader import config, load_yaml, load_txt
from textblock_formatter import build_courses_block, build_locations_block
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

llm: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None
system_prompt: str = ""

CONFIG_PATH = Path(config["paths"]["config"])
SYSTEM_PROMPT_PATH = Path(config["paths"]["system_prompt"])

def make_instruction() -> str:
    """Создание инструкции для LLM."""
    courses = load_yaml(Path(config["paths"]["courses"]))
    locations = load_yaml(Path(config["paths"]["locations"]))
    return (
        "Инструкция:\n" +
        system_prompt + "\n" +
        build_courses_block(courses) + "\n" +
        build_locations_block(locations) + "\n"
    )
    
def load_sampling_params():
    global sampling_params
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sampling_params = SamplingParams(
        temperature=cfg["sampling"].get("temperature", 0.3),
        top_p=cfg["sampling"].get("top_p", 0.5),
        max_tokens=cfg["sampling"].get("max_tokens", 350),
        stop=cfg["sampling"].get("stop", "Пользователь:"),
    )

def load_system_prompt():
    global system_prompt
    system_prompt = load_txt(SYSTEM_PROMPT_PATH)

class ConfigWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH.name):
            load_sampling_params()
        elif event.src_path.endswith(SYSTEM_PROMPT_PATH.name):
            load_system_prompt()

@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения."""
    global llm
    llm = LLM(
        model=config["model"].get("name","PrunaAI/IlyaGusev-saiga_mistral_7b_merged-AWQ-4bit-smashed"),
        quantization=config["model"].get("quantization", "awq_marlin"),
        gpu_memory_utilization=config["model"].get("gpu_memory_utilization", "0.6"),
        dtype=config["model"].get("dtype", "auto"),
        max_model_len=config["model"].get("max_model_len", 5000),
    )
    # Запускаем watchdog
    event_handler = ConfigWatcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()

    yield  # приложение работает

    observer.stop()
    observer.join()

def get_llm_reply(context: list) -> str:
    prompt_parts = [make_instruction()]
    prompt_parts.append(f"Диалог с пользователем:")
    for msg in context:
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "user":
            prompt_parts.append(f"Пользователь: {content}")
        elif role == "bot":
            prompt_parts.append(f"Бот: {content}")

    prompt_parts.append("Бот:")

    prompt = "\n".join(prompt_parts)
    print(f"Sampling params: {sampling_params}")
    print(f"Generated prompt: {prompt}")
    
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text

    print(f"Generated text: {generated_text}")

    cleaned_text = re.sub(r"\s*бот:|\s*bot:", "", generated_text, flags=re.IGNORECASE)
    cleaned_text = re.split(r"\s*пользователь:", cleaned_text, flags=re.IGNORECASE)[0]
    return cleaned_text.strip()