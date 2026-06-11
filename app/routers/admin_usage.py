from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DbSession

from app.core.database import get_db
from app.dependencies.auth import require_role
from app.models.user import User
from app.schemas.admin_usage import DailyUsageOut
from app.schemas.user import AppRole
from app.services.admin_usage_service import AdminUsageService

router = APIRouter(prefix="/admin/usage", tags=["admin"])


@router.get("/daily", response_model=list[DailyUsageOut])
def get_daily_usage(
    days: int = Query(default=30, ge=1, le=90),
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return AdminUsageService(db).get_daily_usage(days, admin_user)
