from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from db import get_db
from deps.auth import get_effective_permissions, require_role
from models.user import User
from models.user_permission import UserPermission
from schemas.user import AdminUserCreateIn, AdminUserUpdateIn, AppRole, UserOut

router = APIRouter(prefix="/admin/users", tags=["admin"])


def _to_user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        tenant_id=user.tenant_id,
        microsoft_oid=user.microsoft_oid,
        role=AppRole(user.role),
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        permissions=sorted(get_effective_permissions(user)),
        input_tokens=user.input_tokens,
        output_tokens=user.output_tokens,
        total_tokens=user.total_tokens,
        token_budget=user.token_budget,
    )


@router.get("", response_model=list[UserOut])
def list_users(
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    users = db.scalars(
        select(User).where(User.tenant_id == admin_user.tenant_id).order_by(User.created_at.desc())
    ).all()
    return [_to_user_out(user) for user in users]


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: AdminUserCreateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    if not admin_user.tenant_id:
        raise HTTPException(status_code=400, detail="Admin tenant is not configured")

    existing = db.scalar(
        select(User).where(
            User.tenant_id == admin_user.tenant_id,
            User.microsoft_oid == payload.microsoft_oid,
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="User already exists for this OID")

    user = User(
        email=payload.email.lower().strip() if payload.email else None,
        tenant_id=admin_user.tenant_id,
        microsoft_oid=payload.microsoft_oid.strip(),
        role=payload.role.value,
        auth_provider="microsoft",
        is_active=True,
        token_budget=payload.token_budget,
    )
    db.add(user)
    db.flush()

    for permission_code in sorted(set(payload.permissions)):
        normalized_permission = permission_code.strip()
        if not normalized_permission:
            continue
        db.add(
            UserPermission(
                user_id=user.id,
                permission_code=normalized_permission,
            )
        )

    db.commit()
    db.refresh(user)
    return _to_user_out(user)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(
    user_id: str,
    payload: AdminUserUpdateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.tenant_id != admin_user.tenant_id:
        raise HTTPException(status_code=403, detail="User belongs to another tenant")

    if payload.role is not None:
        user.role = payload.role.value
    if payload.is_active is not None:
        user.is_active = payload.is_active

    if payload.permissions is not None:
        db.query(UserPermission).where(UserPermission.user_id == user.id).delete()
        for permission_code in sorted(set(payload.permissions)):
            normalized_permission = permission_code.strip()
            if not normalized_permission:
                continue
            db.add(
                UserPermission(
                    user_id=user.id,
                    permission_code=normalized_permission,
                )
            )
    if payload.token_budget is not None:
        user.token_budget = payload.token_budget

    db.commit()
    db.refresh(user)
    return _to_user_out(user)
