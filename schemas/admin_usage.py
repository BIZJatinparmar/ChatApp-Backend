from pydantic import BaseModel


class DailyUsageOut(BaseModel):
    date: str
    active_users: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
