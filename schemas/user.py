from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class AppRole(str, Enum):
    admin = "admin"
    user = "user"


class UserOut(BaseModel):
    id: str
    email: EmailStr | None
    tenant_id: str | None = None
    microsoft_oid: str | None = None
    role: AppRole
    auth_provider: str
    is_active: bool
    permissions: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_budget: int = 0


class AdminUserCreateIn(BaseModel):
    microsoft_oid: str = Field(min_length=1, max_length=64)
    email: EmailStr | None = None
    role: AppRole = AppRole.user
    permissions: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=0, ge=0)


class AdminUserUpdateIn(BaseModel):
    role: AppRole | None = None
    is_active: bool | None = None
    permissions: list[str] | None = None
    token_budget: int | None = Field(default=None, ge=0)
