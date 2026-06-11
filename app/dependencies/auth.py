from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session as DbSession

from app.core.database import get_db
from app.models.session import Session
from app.models.user import User
from app.security.sessions import SESSION_COOKIE_NAME

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "user": {
        "chat:use",
        "conversation:read",
        "conversation:write",
        "message:write",
        "files:upload",
    },
}


def get_current_user(
    db: DbSession = Depends(get_db),
    session_id: Annotated[str | None, Cookie(
        alias=SESSION_COOKIE_NAME)] = None,
) -> User:
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    sess = db.get(Session, session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    expires_at = sess.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < datetime.now(timezone.utc):
        db.delete(sess)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    user = db.get(User, sess.user_id)
    if not user:
        db.delete(sess)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive")

    return user


def get_effective_permissions(user: User) -> set[str]:
    role_permissions = ROLE_PERMISSIONS.get(user.role, set())
    if "*" in role_permissions:
        return {"*"}
    explicit_permissions = {
        permission.permission_code for permission in user.permissions
    }
    return role_permissions | explicit_permissions


def require_role(required_role: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}",
            )
        return user

    return dependency


def require_permissions(*required_permissions: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        effective_permissions = get_effective_permissions(user)
        if "*" in effective_permissions:
            return user
        missing = [p for p in required_permissions if p not in effective_permissions]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing)}",
            )
        return user

    return dependency
