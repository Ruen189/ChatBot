from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """Сообщение в диалоге."""
    role: Literal["user", "bot"]
    content: str = Field(..., min_length=1, max_length=1000)


class GenerateRequest(BaseModel):
    """Модель входных данных для генерации ответа."""
    user_input: str = Field(..., min_length=1, max_length=1000)
    context: List[Message] = Field(default_factory=list)

    @field_validator("context")
    def truncate_context_length(cls, messages: List[Message]):
        """Обрезает контекст, чтобы суммарная длина была <= 1000 символов."""
        max_total_length = 1000

        # Обрезаем отдельные сообщения до 1000 символов
        for msg in messages:
            if len(msg.content) > 1000:
                msg.content = msg.content[-1000:]

        total_length = sum(len(msg.content) for msg in messages)
        if total_length <= max_total_length:
            return messages

        truncated_context = []
        length_accum = 0

        # Берём последние сообщения, пока не достигнем лимита
        for msg in reversed(messages):
            msg_len = len(msg.content)
            if length_accum + msg_len > max_total_length:
                allowed_len = max_total_length - length_accum
                if allowed_len > 0:
                    msg.content = msg.content[-allowed_len:]
                    truncated_context.append(msg)
                break
            truncated_context.append(msg)
            length_accum += msg_len

        truncated_context.reverse()
        return truncated_context
