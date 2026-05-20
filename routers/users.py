from fastapi import APIRouter, Depends

from deps.auth import get_current_user
from models.user import User
from schemas.user import UserOut

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(
        id=user.id,
        email=user.email,
        input_tokens=user.input_tokens,
        output_tokens=user.output_tokens,
        total_tokens=user.total_tokens,
    )
