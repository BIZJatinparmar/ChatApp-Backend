from fastapi import APIRouter, Depends

from deps.auth import get_current_user
from models.User import User
from schemas.user import UserOut

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email)