from typing import List
from pydantic import BaseModel, Field, field_validator

class GenerateRequest(BaseModel):
    """Модель входных данных для генерации ответа."""
    user_input: str = Field(..., min_length=1, max_length=1000)
    context: List[str] = Field(default_factory=list)

    @field_validator("context")
    def truncate_context_length(cls, v):
        if not isinstance(v, list):
            raise TypeError("context must be a list")

        max_total_length = 2000
        v = [s if len(s) <= 1000 else s[-1000:] for s in v]

        total_length = sum(len(s) for s in v)
        if total_length <= max_total_length:
            return v

        truncated_context = []
        length_accum = 0
        for s in reversed(v):
            if length_accum + len(s) > max_total_length:
                allowed_len = max_total_length - length_accum
                if allowed_len > 0:
                    truncated_context.append(s[-allowed_len:])
                break
            truncated_context.append(s)
            length_accum += len(s)

        truncated_context.reverse()
        return truncated_context
