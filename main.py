from fastapi import FastAPI
from pydantic import BaseModel
import re
#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch
import uvicorn
import json
from pathlib import Path

COURSES_PATH = Path(__file__).parent / "courses.json"
with open(COURSES_PATH, encoding="utf-8") as f:
    COURSES = json.load(f)

print(COURSES)
app = FastAPI()

"""
# Загружаем модель (квантованную bitsandbytes)
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    device_map="auto",             # сам расшлёт слои по доступным GPU/CPU
    torch_dtype=torch.float16      # половинная точность для экономии VRAM
)
"""
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

def age_reply(age):
    matching = find_courses_by_age(age)
    count = len(matching)
    if count == 0:
        return "К сожалению, для данного возраста курсов не найдено."

    # Выбираем префикс для единственного/множественного числа
    if count == 1:
        header = "Для вашего возраста был найден подходящий курс:"
    else:
        header = "Для вашего возраста были найдены подходящие курсы:"

    # Формируем список описаний
    lines = [header]
    for c in matching:
        lines.append(f"\n• {c['title']} — {c['description']} ({c['url']})")
    return "".join(lines)

def directions_reply(directions):
    dir_ = directions[0]
    available = [c["id"] for c in COURSES]
    if dir_ in available:
        title = next(c['title'] for c in COURSES if c['id'] == dir_)
        return f"Курс «{title}» доступен для любого возраста от {next(c['min_age'] for c in COURSES if c['id'] == dir_)} лет."
    return "К сожалению, такого направления нет. Уточните возраст или выберите одно из наших направлений."

def exam_reply():
    exam_course = next((c for c in COURSES if c['id'] == 'exam'), None)
    if exam_course:
        return (
            f"Рекомендуем курс подготовки к экзаменам: {exam_course['title']} — "
            f"{exam_course['description']} ({exam_course['url']})"
        )

def rule_based_reply(data):
    age = data.get("age")
    if data.get("multiple"):
        return "Уточните, пожалуйста, возраст или имена каждого ребёнка."

    if age is not None:
        return age_reply(age)

    directions = data.get("directions") or []
    if directions:
        return directions_reply(directions)

    if data.get("exam"):
        return exam_reply()

    return None

"""
def get_llm_reply(user_input: str, context: list[str],
                  max_new_tokens: int = 128) -> str:
    prompt = "\n".join(context + [f"Пользователь: {user_input}", "Бот:"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # отрезаем до ответа
    return text.split("Бот:")[-1].strip()
"""


@app.post("/generate")
def generate(req: Request):
    data = parse_user(req.user_input)
    reply = rule_based_reply(data)
    if reply:
        return {"reply": reply}
    # fallback на LLM
    # bot_reply = get_llm_reply(req.user_input, req.context)
    bot_reply = "Тут должна быть LLM"
    return {"reply": bot_reply}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)




