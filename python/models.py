"""Модуль для определения моделей данных, используемых в приложении."""
from typing import List, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Сообщение в диалоге."""
    role: Literal["user", "bot"]
    content: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    """Модель входных данных для генерации ответа."""
    context: List[Message] = Field(default_factory=list)