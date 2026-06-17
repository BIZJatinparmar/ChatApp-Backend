from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from app.models.user import User
from app.repositories.usage_repository import UsageRepository
from app.repositories.user_permission_repository import UserPermissionRepository
from app.repositories.user_repository import UserRepository
from app.schemas.user import AdminUserCreateIn, AdminUserUpdateIn, UserOut
from app.services.user_service import to_user_out


class AdminUserService:
    def __init__(self, db: DbSession):
        self.db = db
        self.users = UserRepository(db)
        self.permissions = UserPermissionRepository(db)
        self.usage = UsageRepository(db)

    def list_users(self, admin_user: User) -> list[UserOut]:
        users = self.users.list_for_tenant(admin_user.tenant_id)
        usage_by_user = self.usage.usage_for_users([user.id for user in users])
        return [to_user_out(user, usage_by_user[user.id]) for user in users]

    def create_user(self, payload: AdminUserCreateIn, admin_user: User) -> UserOut:
        if not admin_user.tenant_id:
            raise HTTPException(status_code=400, detail="Admin tenant is not configured")

        existing = self.users.get_by_tenant_oid(admin_user.tenant_id, payload.microsoft_oid)
        if existing:
            raise HTTPException(status_code=409, detail="User already exists for this OID")

        user = self.users.create(
            User(
                email=payload.email.lower().strip() if payload.email else None,
                tenant_id=admin_user.tenant_id,
                microsoft_oid=payload.microsoft_oid.strip(),
                role=payload.role.value,
                auth_provider="microsoft",
                is_active=True,
                token_budget=payload.token_budget,
            )
        )
        self.permissions.replace_for_user(user.id, payload.permissions)
        self.db.commit()
        self.db.refresh(user)
        return to_user_out(user, self.usage.usage_for_user(user.id))

    def update_user(self, user_id: str, payload: AdminUserUpdateIn, admin_user: User) -> UserOut:
        user = self.users.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.tenant_id != admin_user.tenant_id:
            raise HTTPException(status_code=403, detail="User belongs to another tenant")

        if payload.role is not None:
            user.role = payload.role.value
        if payload.is_active is not None:
            user.is_active = payload.is_active
        if payload.permissions is not None:
            self.permissions.replace_for_user(user.id, payload.permissions)
        if payload.token_budget is not None:
            user.token_budget = payload.token_budget

        self.db.commit()
        self.db.refresh(user)
        return to_user_out(user, self.usage.usage_for_user(user.id))
