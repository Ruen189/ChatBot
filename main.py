from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from vllm import LLM, SamplingParams
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
import json
import yaml
import logging


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

API_KEYS = set(config.get("api_keys", []))


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_json_data(path: Path) -> list:
    if not path.exists():
        logging.error(f"JSON data file not found: {path}")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logging.error(f"JSON data in {path} is not a list")
            return []
        return data
    except json.JSONDecodeError:
        logging.error(f"JSON decode error in file: {path}")
        return []


COURSES = load_json_data(Path(config["paths"]["courses"]))
LOCATIONS = load_json_data(Path(config["paths"]["locations"]))


def build_courses_block(courses: list) -> str:
    if not courses:
        return "Информация про курсы в Real-IT отсутствует."
    return "Информация про курсы в Real-IT:\n" + "\n".join(
        f"- {c.get('title', 'Без названия')} (от {c.get('min_age', '?')} до {c.get('max_age', '?')} лет): "
        f"{c.get('description', '')} ({c.get('url', '')})"
        for c in courses
    )


def build_locations_block(locations: list) -> str:
    if not locations:
        return "Информация про филиалы Real-IT отсутствует."
    return "Информация про филиалы Real-IT только в Екатеринбурге:\n" + "\n".join(
        f"- {l.get('title', 'Без названия')} ({l.get('street', '')}): "
        f"{l.get('entrance', '')} ({l.get('url', '')})."
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
    user_input: str = Field(..., min_length=1, max_length=1000)
    context: List[str] = []

    @field_validator("context")
    def check_context_length(cls, v):
        if not isinstance(v, list):
            raise TypeError("context must be a list")
        for s in v:
            if len(s) > 1000:
                raise ValueError("Each context string must be at most 1000 characters")
        return v


def get_llm_reply(user_input: str, context: List[str],
                  max_new_tokens: Optional[int] = None) -> str:
    prompt = "\n".join([instruction] + context +
                       [f"Пользователь: {user_input}", "Бот:"])
    sampling_params = SamplingParams(
        temperature=config["sampling"].get("temperature", 0.3),
        top_p=config["sampling"].get("top_p", 0.5),
        max_tokens=max_new_tokens or config["sampling"].get("max_tokens", 350),
    )
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    return generated_text.rsplit("Бот:", 1)[-1].strip()


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
