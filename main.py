import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from vllm import LLM, SamplingParams


def load_config(path: str = "config.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
API_KEYS = set(config.get("api_keys", []))


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Проверка API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_yaml_data(path: Path) -> dict:
    """Загрузка и проверка данных из YAML файла."""
    if not path.exists():
        logging.error(f"YAML data file not found: {path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logging.error(f"YAML data in {path} is not a dict (ожидался словарь городов)")
            return {}
        return data
    except yaml.YAMLError as e:
        logging.error(f"YAML decode error in file {path}: {e}")
        return {}


COURSES = load_yaml_data(Path(config["paths"]["courses"]))
LOCATIONS = load_yaml_data(Path(config["paths"]["locations"]))


def build_courses_block(courses_data: dict) -> str:
    """Формирование текстового блока с курсами."""
    courses = courses_data.get("courses", [])
    if not courses:
        return "Информация про курсы в Real-IT отсутствует."

    return "Информация про курсы в Real-IT:\n" + "\n".join(
        f"- {c.get('title', 'Без названия')} "
        f"(от {c.get('min_age', '?')} до {c.get('max_age', '?')} лет): "
        f"{c.get('description', '')} ({c.get('url', '')})"
        for c in courses
    )


def build_locations_block(locations: dict, city: str = "Екатеринбург") -> str:
    """Формирование текстового блока с филиалами."""
    city_locations = locations.get(city, [])
    if not city_locations:
        return f"Информация про филиалы Real-IT в городе {city} отсутствует."

    return f"Информация про филиалы Real-IT в городе {city}:\n" + "\n".join(
        f"- {l.get('title', 'Без названия')} ({l.get('street', '')}): "
        f"{l.get('entrance', '')}."
        for l in city_locations
    )


instruction = (
    config["instruction"]["system_prompt"] + "\n" +
    build_courses_block(COURSES) + "\n" +
    build_locations_block(LOCATIONS) + "\n\n"
)

llm: Optional[LLM] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
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


app = FastAPI(lifespan=lifespan)

allow_origins = config.get("cors", {}).get("allow_origins", [])
if not allow_origins or allow_origins == ["*"]:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


class GenerateRequest(BaseModel):
    """Модель входных данных для генерации ответа."""
    user_input: str = Field(..., min_length=1, max_length=1000)
    context: List[str] = Field(default_factory=list)

    @field_validator("context")
    def truncate_context_length(cls, v):
        if not isinstance(v, list):
            raise TypeError("context must be a list")

        max_total_length = 2000
        v = [s if len(s) <= 1000 else s[-1000:] for s in v]

        total_length = sum(len(s) for s in v)
        if total_length <= max_total_length:
            return v

        truncated_context = []
        length_accum = 0
        for s in reversed(v):
            if length_accum + len(s) > max_total_length:
                allowed_len = max_total_length - length_accum
                if allowed_len > 0:
                    truncated_context.append(s[-allowed_len:])
                break
            truncated_context.append(s)
            length_accum += len(s)

        truncated_context.reverse()
        return truncated_context


sampling_params = SamplingParams(
    temperature=config["sampling"].get("temperature", 0.3),
    top_p=config["sampling"].get("top_p", 0.5),
    max_tokens=config["sampling"].get("max_tokens", 350),
)


def get_llm_reply(user_input: str, context: List[str]) -> str:
    """Формирование промпта и получение ответа от модели."""
    prompt = "\n".join(
        [instruction] + context + [f"Пользователь: {user_input}", "Бот:"]
    )
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    cleaned_text = re.split(r"\s*bot:|\s*бот:", generated_text, flags=re.IGNORECASE)[0]
    return cleaned_text.strip()


@app.post("/generate")
async def generate(req: GenerateRequest, x_api_key: str = Header(None)):
    await verify_api_key(x_api_key)
    reply = get_llm_reply(req.user_input, req.context)
    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["uvicorn"].get("host", "0.0.0.0"),
        port=config["uvicorn"].get("port", 8000),
        reload=config["uvicorn"].get("reload", False),
        workers=config["uvicorn"].get("workers", 1),
        log_level=config["uvicorn"].get("log_level", "info"),
    )
