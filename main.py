from fastapi import FastAPI
from pydantic import BaseModel
import re
from vllm import LLM, SamplingParams
import uvicorn
import json
from pathlib import Path

COURSES_PATH = Path(__file__).parent / "courses.json"
with open(COURSES_PATH, encoding="utf-8") as f:
    COURSES = json.load(f)
app = FastAPI()

llm = LLM(
    model="cybrtooth/TheBloke-Mistral-7B-Instruct-v0.2-GGUF",
    quantization="awq",
    gpu_memory_utilization=0.7, # или даже меньше
    dtype = "float16",
    max_model_len=3072,
)

course_info_block = "\n".join(
        f"- {c['title']} (от {c['min_age']} до {c['max_age']} лет): {c['description']} ({c['url']})"  # Добавить параметр который может потребовать прохождение предидущих курсов
        for c in COURSES
    )

class Request(BaseModel):
    user_input: str
    context: list[str] = []

def get_llm_reply(user_input: str, context: list[str]  , max_new_tokens: int = 1024) -> str:
    # Преобразуем курсы в читаемый текст
    instruction = (
        "Ты — ассистент, который помогает выбрать подходящий курс из списка ниже."
        "Ты отвечаешь на прямую пользователю, поэтому используй дружелюбный и вежливый тон."
        "Отвечай на русском языке НЕ ИСПОЛЬЗУЙ английский язык ни при каких обстоятельствах."
        "Не упоминай курсы, которые не подходят по запросу пользователя. Из ответа исключи фразы вроде 'Пользователь сообщил' и 'я рекомендую курс'."
        "Если информации недостаточно, попроси уточнить возраст или направление."
        "Если по возрасту подходит несколько курсов, перечисли их все."
        "Если сообщение пользователя не содержит смысла, НЕ ОТВЕЧАЙ на него и НЕ ПЫТАЙСЯ ИНТЕРПРЕТИРОВАТЬ его."
        "Если нашёл подходящие курсы, перечисли их в ответе и указывай полную информацию с ссылками."
        "Выбирай подходящие курсы из приведённых ниже, основываясь на возрасте пользователя, и направлениях, которые он укажет."
        f"{course_info_block}\n\n"
        )

    prompt = "\n".join(
        [instruction] + context[-1:] + [f"Пользователь: {user_input}", "Бот:"]
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=max_new_tokens,
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
    uvicorn.run("main:app", host="192.168.0.116", reload=True)




