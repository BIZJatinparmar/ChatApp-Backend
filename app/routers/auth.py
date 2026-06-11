from __future__ import annotations

from fastapi import APIRouter, Depends, Response, Cookie
from sqlalchemy.orm import Session as DbSession
from typing import Annotated

from app.security.sessions import SESSION_COOKIE_NAME
from app.core.database import get_db
from app.schemas.auth import MicrosoftLoginIn
from app.schemas.user import UserOut
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/microsoft/login", response_model=UserOut)
def microsoft_login(
    payload: MicrosoftLoginIn,
    response: Response,
    db: DbSession = Depends(get_db),
):
    return AuthService(db).microsoft_login(payload, response)


@router.post("/logout")
def logout(
    response: Response,
    db: DbSession = Depends(get_db),
    session_id: Annotated[str | None, Cookie(
        alias=SESSION_COOKIE_NAME)] = None,
):
    return AuthService(db).logout(response, session_id)
