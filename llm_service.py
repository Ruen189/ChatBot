import re
from contextlib import asynccontextmanager
from typing import Optional, List
from vllm import LLM, SamplingParams
from config import config
from textblock_formatter import build_courses_block, build_locations_block
from data_loader import load_yaml_data
from pathlib import Path


COURSES = load_yaml_data(Path(config["paths"]["courses"]))
LOCATIONS = load_yaml_data(Path(config["paths"]["locations"]))

instruction = (
    "Инструкция для LLM:\n" +
    config["instruction"]["system_prompt"] + "\n" +
    build_courses_block(COURSES) + "\n" +
    build_locations_block(LOCATIONS) + "\n\n"
)
llm: Optional[LLM] = None

@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения."""
    global llm
    llm = LLM(
        model=config["model"]["name"],
        quantization=config["model"].get("quantization"),
        gpu_memory_utilization=config["model"].get("gpu_memory_utilization"),
        dtype=config["model"].get("dtype"),
        max_model_len=config["model"].get("max_model_len"),
    )
    yield

sampling_params = SamplingParams(
    temperature=config["sampling"].get("temperature", 0.3),
    top_p=config["sampling"].get("top_p", 0.5),
    max_tokens=config["sampling"].get("max_tokens", 350),
    stop=config["sampling"].get("stop", "Пользователь:"),
)

def get_llm_reply(context: list) -> str:
    prompt_parts = [instruction] 
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
    
    print(f"Generated prompt: {prompt}")

    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text

    print(f"Generated text: {generated_text}")

    cleaned_text = re.sub(r"\s*бот:|\s*bot:", "", generated_text, flags=re.IGNORECASE)
    cleaned_text = re.split(r"\s*пользователь:", cleaned_text, flags=re.IGNORECASE)[0]
    return cleaned_text.strip()