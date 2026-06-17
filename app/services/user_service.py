from app.dependencies.auth import get_effective_permissions
from app.models.user import User
from app.repositories.usage_repository import UsageRepository, UsageTotals
from app.schemas.user import AppRole, UserOut


def to_user_out(user: User, usage: UsageTotals | None = None) -> UserOut:
    usage = usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return UserOut(
        id=user.id,
        email=user.email,
        tenant_id=user.tenant_id,
        microsoft_oid=user.microsoft_oid,
        role=AppRole(user.role),
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        permissions=sorted(get_effective_permissions(user)),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        token_budget=user.token_budget,
    )


class UserService:
    def __init__(self, db):
        self.usage = UsageRepository(db)

    def get_current_user_profile(self, user: User) -> UserOut:
        return to_user_out(user, self.usage.usage_for_user(user.id))
