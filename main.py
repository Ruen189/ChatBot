import re
import json
import uuid
import logging
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from python.models import GenerateRequest
from python.loader import config, verify_api_key
from python.llm_service import lifespan, get_llm_reply

# Настройка логирования
logging.basicConfig(
    filename="requests.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

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

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request, x_api_key: str = Header(None)):
    await verify_api_key(x_api_key)
    
    request_id = f"req-{uuid.uuid4().hex}"
    client_ip = request.client.host if request.client else "unknown"

    # Берем последнее сообщение пользователя
    last_user_message = next((msg.content for msg in reversed(req.context) if msg.role == "user"), "")

    async def stream():
        previous_text = ""
        full_response = ""  # собираем весь ответ

        async for chunk in get_llm_reply(req.context, request_id=request_id):
            cleaned_chunk = re.sub(r'[ \t]+', ' ', chunk.strip())

            if cleaned_chunk.startswith(previous_text):
                new_part = cleaned_chunk[len(previous_text):]
            else:
                new_part = cleaned_chunk

            previous_text = cleaned_chunk

            if new_part:
                full_response += new_part
                yield json.dumps({"text": new_part}, ensure_ascii=False) + "\n"

        # Логируем IP, последнее сообщение пользователя и полный ответ бота в читаемом формате
        logger.info(
            f"IP: {client_ip} | UserMessage: {last_user_message} | BotResponse: {full_response}"
        )
        yield json.dumps({"done": True}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["uvicorn"].get("host", "0.0.0.0"),
        port=config["uvicorn"].get("port", 8000),
        reload=config["uvicorn"].get("reload", False),
        workers=config["uvicorn"].get("workers", 1),
        log_level=config["uvicorn"].get("log_level", "info"),
    )
