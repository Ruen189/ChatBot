import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import re
from python.loader import config
from python.security import verify_api_key
from python.llm_service import lifespan, get_llm_reply
from fastapi.responses import StreamingResponse
import json
from python.models import GenerateRequest


log_cfg = config.get("logging", {})
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO")),
    format=log_cfg.get("format", "%(asctime)s %(levelname)s %(message)s"),
    filename=log_cfg.get("filename"),
    filemode=log_cfg.get("filemode", "a"),
)
logger = logging.getLogger(__name__)


app = FastAPI(lifespan=lifespan)

allow_origins = config.get("cors", {}).get("allow_origins", [])
if not allow_origins or allow_origins == ["*"]:
    allow_origins = ["*"]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body_bytes = await request.body()
    body_text = body_bytes.decode('utf-8') if body_bytes else "EMPTY"

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.2f}ms "
        f"Body: {body_text}"
    )

    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

@app.post("/generate")
async def generate(req: GenerateRequest, x_api_key: str = Header(None)):
    await verify_api_key(x_api_key)

    async def stream():
        previous_text = ""
        async for chunk in get_llm_reply(req.context):
            cleaned_chunk = re.sub(r'[ \t]+', ' ', chunk.strip())


            if cleaned_chunk.startswith(previous_text):
                new_part = cleaned_chunk[len(previous_text):]
            else:
                new_part = cleaned_chunk

            previous_text = cleaned_chunk

            if new_part:
                yield json.dumps({"text": new_part}, ensure_ascii=False) + "\n"

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
