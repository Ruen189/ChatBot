from fastapi import FastAPI
from pydantic import BaseModel
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

print("CUDA available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
app = FastAPI()

# Загружаем модель (квантованную bitsandbytes)
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    device_map="auto",             # сам расшлёт слои по доступным GPU/CPU
    torch_dtype=torch.float16      # половинная точность для экономии VRAM
)

class Request(BaseModel):
    user_input: str
    context: list[str] = []

def parse_user(msg: str):
    data = {}
    age_m = re.search(r"\b(\d{1,2})\s*(лет|года|год)\b", msg)
    data["age"] = int(age_m.group(1)) if age_m else None
    data["exam"] = bool(re.search(r"экзамен", msg, re.I))
    data["directions"] = re.findall(r"направление\s+(\w+)", msg, re.I)
    data["multiple"] = bool(re.search(r"\b(двое|трое|два|три) детей\b", msg))
    return data

def rule_based_reply(data):
    age = data["age"]
    if data["multiple"]:
        return "Уточните, пожалуйста, возраст или имена каждого ребёнка."
    if age is not None:
        if age < 7:
            return "Рекомендую погулять пару лет — в этом возрасте мы ещё не начинаем обучение.\nЕсть ли у ребёнка опыт работы с компьютером?"
        if age < 12:
            return "Вот несколько курсов для детей 7–11 лет: ...\nБыл ли опыт работы с компьютером?"
        if age <= 18:
            return f"Курс по возрасту {age} лет: «Алгоритмы и логика»."
        return "Для взрослых у нас есть Курсы Python, Data Science и др."
    if data["directions"]:
        dir_ = data["directions"][0]
        # проверяем, есть ли такое направление
        available = ["python", "web", "game"]
        if dir_ in available:
            return f"Курс «{dir_.title()}» доступен для любого возраста от 7 лет."
        else:
            return "К сожалению, такого направления нет. Уточните возраст или выберите одно из наших направлений."
    return None

@app.post("/generate")
def generate(req: Request):
    data = parse_user(req.user_input)
    reply = rule_based_reply(data)
    if reply:
        return { "reply": reply }
    # fallback на LLM
    prompt = "\n".join(req.context + [f"Пользователь: {req.user_input}", "Бот:"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=128)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # отрезаем до ответа
    bot_reply = text.split("Бот:")[-1].strip()
    return { "reply": bot_reply }

if __name__ == "__main__":
    print(torch.version.cuda)
    uvicorn.run("main:app", reload=True)




