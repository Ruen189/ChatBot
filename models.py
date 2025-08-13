from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """Сообщение в диалоге."""
    role: Literal["user", "bot"]
    content: str = Field(..., min_length=1)  # Нет max_length


class GenerateRequest(BaseModel):
    """Модель входных данных для генерации ответа."""
    user_input: str = Field(..., min_length=1)  # Нет max_length
    context: List[Message] = Field(default_factory=list)