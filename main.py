from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
import json
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
API_KEYS = set(config.get("api_keys", []))


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Проверяет наличие и корректность API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_json_data(path: Path) -> list:
    """Загружает данные из JSON файла."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


COURSES = load_json_data(Path(config["paths"]["courses"]))
LOCATIONS = load_json_data(Path(config["paths"]["locations"]))


def build_courses_block(courses: list) -> str:
    """Формирует текстовый блок с информацией о курсах."""
    return "Информация про курсы в Real-IT:\n" + "\n".join(
        f"- {c['title']} (от {c['min_age']} до {c['max_age']} лет): "
        f"{c['description']} ({c['url']})"
        for c in courses
    )


def build_locations_block(locations: list) -> str:
    """Формирует текстовый блок с информацией о филиалах."""
    return "Информация про филиалы Real-IT только в Екатеринбурге:\n" + "\n".join(
        f"- {l['title']} ({l['street']}): {l['entrance']} ({l['url']})."
        for l in locations
    )


instruction = (
    config["instruction"]["system_prompt"] + "\n" +
    build_courses_block(COURSES) + "\n" +
    build_locations_block(LOCATIONS) + "\n\n"
)

llm: Optional[LLM] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация LLM при запуске приложения."""
    global llm
    llm = LLM(
        model=config["model"]["name"],
        quantization=config["model"]["quantization"],
        gpu_memory_utilization=config["model"]["gpu_memory_utilization"],
        dtype=config["model"]["dtype"],
        max_model_len=config["model"]["max_model_len"],
    )
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    """Модель запроса для генерации ответа."""
    user_input: str
    context: List[str] = []


def get_llm_reply(user_input: str, context: List[str],
                  max_new_tokens: Optional[int] = None) -> str:
    """Генерирует ответ от модели."""
    prompt = "\n".join([instruction] + context +
                       [f"Пользователь: {user_input}", "Бот:"])
    sampling_params = SamplingParams(
        temperature=config["sampling"]["temperature"],
        top_p=config["sampling"]["top_p"],
        max_tokens=max_new_tokens or config["sampling"]["max_tokens"],
    )
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    return generated_text.rsplit("Бот:", 1)[-1].strip()


@app.post("/generate")
async def generate(req: GenerateRequest, x_api_key: str = Header(None)):
    """Эндпоинт генерации ответа модели."""
    await verify_api_key(x_api_key)
    reply = get_llm_reply(req.user_input, req.context)
    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["uvicorn"].get("host", "127.0.0.1"),
        port=config["uvicorn"].get("port", 8000),
        reload=config["uvicorn"].get("reload", False),
        workers=config["uvicorn"].get("workers", 1),
        log_level=config["uvicorn"].get("log_level", "info"),
    )
