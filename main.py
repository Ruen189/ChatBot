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
    model="Qwen/Qwen3-4B-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.7, # или даже меньше
    dtype = "auto",
    max_model_len=2048,
)

class Request(BaseModel):
    user_input: str
    context: list[str] = []


def parse_user(msg: str):
    data = {}
    age_m = re.search(r"\b(\d{1,2})\s*(лет|года|год|годиков)\b", msg)
    data["age"] = int(age_m.group(1)) if age_m else None
    data["exam"] = bool(re.search(r"экзамен", msg, re.I))
    data["directions"] = re.findall(r"направление\s+(\w+)", msg, re.I)
    data["multiple"] = bool(re.search(r"\b(двое|трое|два|три) детей\b", msg))
    return data

def find_courses_by_age(age: int) -> list[dict]:
    """Возвращает список курсов, подходящих по возрасту."""
    return [
        course for course in COURSES
        if course["min_age"] <= age <= course.get("max_age", age)
    ]

def get_llm_reply(user_input: str, context: list[str]  , max_new_tokens: int = 512) -> str:
    # Преобразуем курсы в читаемый текст
    course_info_block = "\n".join(
        f"- {c['title']} (от {c['min_age']} до {c['max_age']} лет): {c['description']} ({c['url']})"
        for c in COURSES
    )

    instruction = (
        "Ты — ассистент, который помогает выбрать подходящий курс из списка ниже."
        "Ты отвечаешь на прямую пользователю, поэтому используй дружелюбный и вежливый тон."
        "отвечай на русском языке НЕ ИСПОЛЬЗУЙ английский язык ни при каких обстоятельствах."
        "Пример вопроса: Хочу записать ребёнка на курс по программированию, ему 10 лет. Пример ответа: вам подойдёт курс «Основы программирования» на нём изучаются основы построения алгоритмов в простейших средах разработки ссылка: https://it-schools.org/year/course/programming-begin/"
        "Выбирай подходящие курсы из приведённых ниже, основываясь на возрасте пользователя, и направлениях, которые он укажет. Если информации недостаточно, попроси уточнить возраст или направление."
        f"{course_info_block}\n\n"
        )

    prompt = "\n".join(
        [instruction] + context[-1:] + [f"Пользователь: {user_input}", "Бот:"]
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=max_new_tokens,
        top_k=40,
        repetition_penalty=1.1,
        n=1,
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
    uvicorn.run("main:app", reload=True)




