from fastapi import APIRouter, Depends

from deps.auth import get_current_user, get_effective_permissions
from models.user import User
from schemas.user import AppRole, UserOut

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
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
