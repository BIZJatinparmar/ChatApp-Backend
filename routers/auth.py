from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response,Cookie
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from typing import Annotated
from security.sessions import SESSION_COOKIE_NAME
from db import get_db
from models.Session import Session
from models.User import User
from schemas.auth import LoginIn, SignupIn
from schemas.user import UserOut
from security.passwords import hash_password, verify_password
from security.sessions import (
    clear_session_cookie,
    compute_expiry,
    new_session_id,
    set_session_cookie,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(payload: SignupIn, db: DbSession = Depends(get_db)):
    email = payload.email.lower().strip()
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    print(payload.password)
    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserOut(id=user.id, email=user.email)


@router.post("/login")
def login(payload: LoginIn, response: Response, db: DbSession = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.scalar(select(User).where(User.email == email))

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    session_id = new_session_id()
    sess = Session(id=session_id, user_id=user.id, expires_at=compute_expiry())

    db.add(sess)
    db.commit()

    set_session_cookie(response, session_id)
    return {"ok": True}




@router.post("/logout")
def logout(
	    response: Response,
	    db: DbSession = Depends(get_db),
	    session_id: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
	):
        if session_id:
             sess = db.get(Session, session_id)
             if sess:
                db.delete(sess) 
                db.commit()

        clear_session_cookie(response)
        return {"ok": True}
