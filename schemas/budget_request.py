from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class BudgetRequestStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class BudgetRequestCreateIn(BaseModel):
    requested_tokens: int = Field(gt=0, le=10_000_000)
    note: str | None = Field(default=None, max_length=1000)


class BudgetRequestAdminUpdateIn(BaseModel):
    status: BudgetRequestStatus
    approved_tokens: int | None = Field(default=None, gt=0, le=10_000_000)
    admin_note: str | None = Field(default=None, max_length=1000)


class BudgetRequestOut(BaseModel):
    id: str
    user_id: str
    user_email: EmailStr | None = None
    requested_tokens: int
    status: BudgetRequestStatus
    note: str | None = None
    admin_note: str | None = None
    decided_by_user_id: str | None = None
    decided_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
