import uvicorn
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from config import config
from security import verify_api_key
from llm_service import lifespan, get_llm_reply
from models import GenerateRequest

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
