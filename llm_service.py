import re
from contextlib import asynccontextmanager
from typing import Optional, List
from vllm import LLM, SamplingParams
from config import config
from textblock_formatter import build_courses_block, build_locations_block
from data_loader import load_yaml_data
from pathlib import Path
from models import Message

COURSES = load_yaml_data(Path(config["paths"]["courses"]))
LOCATIONS = load_yaml_data(Path(config["paths"]["locations"]))

instruction = (
    config["instruction"]["system_prompt"] + "\n" +
    build_courses_block(COURSES) + "\n" +
    build_locations_block(LOCATIONS) + "\n\n"
)
print(f"Instruction: {instruction}")
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
)

def get_llm_reply(user_input: str, context: List[dict]) -> str:
    prompt_parts = [instruction]

    for msg in context:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            prompt_parts.append(f"Пользователь: {content}")
        elif role == "bot":
            prompt_parts.append(f"Бот: {content}")

    prompt_parts.append(f"Пользователь: {user_input}")
    prompt_parts.append("Бот:")

    prompt = "\n".join(prompt_parts)

    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    
    
    cleaned_text = re.split(r"\s*bot:|\s*бот:|\s*пользователь:", generated_text, flags=re.IGNORECASE)[0]
    return cleaned_text.strip()
