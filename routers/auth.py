from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, Cookie
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from typing import Annotated

from models.session import Session
from models.user import User
from deps.auth import get_effective_permissions
from security.sessions import SESSION_COOKIE_NAME
from db import get_db
from schemas.auth import MicrosoftLoginIn
from schemas.user import AppRole
from schemas.user import UserOut
from security.microsoft_identity import (
    MicrosoftIdentityError,
    parse_admin_oids,
    validate_microsoft_id_token,
)
from security.sessions import (
    clear_session_cookie,
    compute_expiry,
    new_session_id,
    set_session_cookie,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/microsoft/login", response_model=UserOut)
def microsoft_login(
    payload: MicrosoftLoginIn,
    response: Response,
    db: DbSession = Depends(get_db),
):
    try:
        claims = validate_microsoft_id_token(payload.id_token)
    except MicrosoftIdentityError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    tenant_id = str(claims["tid"]).strip()
    microsoft_oid = str(claims["oid"]).strip()
    email_claim = claims.get("email") or claims.get("preferred_username")
    email = email_claim.lower().strip() if isinstance(email_claim, str) else None

    user = db.scalar(
        select(User).where(
            User.tenant_id == tenant_id,
            User.microsoft_oid == microsoft_oid,
        )
    )

    if user is None:
        admin_oids = parse_admin_oids()
        if microsoft_oid not in admin_oids:
            raise HTTPException(
                status_code=403,
                detail="User is not registered. Ask admin to create your account.",
            )
        user = User(
            tenant_id=tenant_id,
            microsoft_oid=microsoft_oid,
            email=email,
            role=AppRole.admin.value,
            auth_provider="microsoft",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    session_id = new_session_id()
    sess = Session(id=session_id, user_id=user.id, expires_at=compute_expiry())

    db.add(sess)
    db.commit()

    set_session_cookie(response, session_id)
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


@router.post("/logout")
def logout(
    response: Response,
    db: DbSession = Depends(get_db),
    session_id: Annotated[str | None, Cookie(
        alias=SESSION_COOKIE_NAME)] = None,
):
    if session_id:
        sess = db.get(Session, session_id)
        if sess:
            db.delete(sess)
            db.commit()

    clear_session_cookie(response)
    return {"ok": True}
