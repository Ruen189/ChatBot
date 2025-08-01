from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import torch
import uvicorn
import json
from pathlib import Path
from typing import List

# Загрузка данных о курсах и локациях
COURSES_PATH = Path(__file__).parent / "courses.json"
with open(COURSES_PATH, encoding="utf-8") as f:
    COURSES = json.load(f)

LOCATIONS_PATH = Path(__file__).parent / "locations.json"
with open(LOCATIONS_PATH, encoding="utf-8") as f:
    LOCATIONS = json.load(f)

# Конфигурация модели
MODEL_NAME = "IlyaGusev/saiga_7b_lora"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент, который помогает выбрать подходящий курс из списка ниже. "
    "Ты отвечаешь на прямую пользователю, поэтому используй дружелюбный и вежливый тон. "
    "Отвечай на русском языке НЕ ИСПОЛЬЗУЙ английский язык ни при каких обстоятельствах. "
    "Не упоминай курсы, которые не подходят по запросу пользователя. "
    "Если информации недостаточно, попроси уточнить возраст или направление."
)

app = FastAPI()

# Инициализация модели и токенизатора
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)


class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            start_token_id=1,
            bot_token_id=9225
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


class Request(BaseModel):
    user_input: str
    context: List[str] = []


def prepare_course_info():
    return "\n".join(
        f"- {c['title']} (от {c['min_age']} до {c['max_age']} лет): {c['description']} ({c['url']})"
        for c in COURSES
    )


def prepare_locations_info():
    return "\n".join(
        f"- {l['title']} ({l['street']}): {l['entrance']} ({l['url']})."
        for l in LOCATIONS
    )


@app.post("/generate")
def generate_response(req: Request):
    course_info = prepare_course_info()
    locations_info = prepare_locations_info()

    system_prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Доступные курсы:\n{course_info}\n\n"
        f"Доступные локации:\n{locations_info}\n\n"
        "Если пользователь спрашивает о локациях или адресах, предоставь информацию из списка локаций."
    )

    conversation = Conversation(system_prompt=system_prompt)

    # Добавляем контекст предыдущих сообщений
    for msg in req.context:
        conversation.add_user_message(msg)

    # Добавляем текущее сообщение пользователя
    conversation.add_user_message(req.user_input)

    prompt = conversation.get_prompt(tokenizer)
    output = generate(model, tokenizer, prompt, generation_config)

    return {"reply": output}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
