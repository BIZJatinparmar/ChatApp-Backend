from app.dependencies.auth import get_effective_permissions
from app.models.user import User
from app.schemas.user import AppRole, UserOut


def to_user_out(user: User) -> UserOut:
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


class UserService:
    def get_current_user_profile(self, user: User) -> UserOut:
        return to_user_out(user)
