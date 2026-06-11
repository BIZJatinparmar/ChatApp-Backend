from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session as DbSession

from app.core.database import get_db
from app.dependencies.auth import require_role
from app.models.user import User
from app.schemas.user import AdminUserCreateIn, AdminUserUpdateIn, AppRole, UserOut
from app.services.admin_user_service import AdminUserService

router = APIRouter(prefix="/admin/users", tags=["admin"])


@router.get("", response_model=list[UserOut])
def list_users(
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return AdminUserService(db).list_users(admin_user)


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: AdminUserCreateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return AdminUserService(db).create_user(payload, admin_user)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(
    user_id: str,
    payload: AdminUserUpdateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return AdminUserService(db).update_user(user_id, payload, admin_user)
