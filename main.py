from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
import json
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

COURSES_PATH = Path(config["paths"]["courses"])
LOCATIONS_PATH = Path(config["paths"]["locations"])

with open(COURSES_PATH, encoding="utf-8") as f:
    COURSES = json.load(f)

with open(LOCATIONS_PATH, encoding="utf-8") as f:
    LOCATIONS = json.load(f)

course_info_block ="Информация про курсы в Real-IT: "+"\n".join(
    f"- {c['title']} (от {c['min_age']} до {c['max_age']} лет): {c['description']} ({c['url']})"
    for c in COURSES
)

locations_block = "Информация про филиалы Real-IT только в Екатеринбурге: "+"\n".join(
    f"- {l['title']} ({l['street']}): {l['entrance']} ({l['url']})."
    for l in LOCATIONS
)

instruction = (
        config["instruction"]["system_prompt"]
        + "\n"
        + course_info_block
        + "\n"
        + locations_block
        + "\n\n"
    )

llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    llm = LLM(
        model=config["model"]["name"],
        quantization=config["model"]["quantization"],
        gpu_memory_utilization=config["model"]["gpu_memory_utilization"],
        dtype=config["model"]["dtype"],
        max_model_len=config["model"]["max_model_len"],
    )
    yield  # здесь приложение продолжит работу
    # здесь можно добавить логику для shutdown, если нужно


app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    user_input: str
    context: list[str] = []

def get_llm_reply(user_input: str, context: list[str], max_new_tokens: int = None) -> str:
    
    prompt = "\n".join(
        [instruction] + context + [f"Пользователь: {user_input}", "Бот:"]
    )

    sampling_params = SamplingParams(
        temperature=config["sampling"]["temperature"],
        top_p=config["sampling"]["top_p"],
        max_tokens=max_new_tokens or config["sampling"]["max_tokens"],
    )

    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    reply = generated_text.split("Бот:")[-1] if "Бот:" in generated_text else generated_text
    return reply.strip()

@app.post("/generate")
def generate(req: Request):
    bot_reply = get_llm_reply(req.user_input, req.context)
    return {"reply": bot_reply}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["uvicorn"].get("host", "127.0.0.1"),
        port=config["uvicorn"].get("port", 8000),
        reload=config["uvicorn"].get("reload", False),
        workers=config["uvicorn"].get("workers", 1),
        log_level=config["uvicorn"].get("log_level", "info"),
    )
