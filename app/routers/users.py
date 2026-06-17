from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from app.core.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.user import UserOut
from app.services.user_service import UserService

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserOut)
def me(
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    return UserService(db).get_current_user_profile(user)
