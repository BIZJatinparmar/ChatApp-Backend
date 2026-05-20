from pydantic import BaseModel, EmailStr


class UserOut(BaseModel):
    id: str
    email: EmailStr | None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
