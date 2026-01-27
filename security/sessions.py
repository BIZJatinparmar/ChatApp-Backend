from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Response

SESSION_COOKIE_NAME = "session"
SESSION_TTL_DAYS = 7


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def compute_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        secure=False,  # True behind HTTPS
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")