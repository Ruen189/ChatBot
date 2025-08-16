from typing import Optional
from fastapi import HTTPException, Header
from .loader import API_KEYS

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Проверка API-ключа."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
