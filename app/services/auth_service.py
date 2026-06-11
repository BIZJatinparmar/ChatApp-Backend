from fastapi import HTTPException, Response
from sqlalchemy.orm import Session as DbSession

from app.models.session import Session
from app.models.user import User
from app.repositories.session_repository import SessionRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import MicrosoftLoginIn
from app.schemas.user import AppRole, UserOut
from app.security.microsoft_identity import (
    MicrosoftIdentityError,
    parse_admin_oids,
    validate_microsoft_id_token,
)
from app.security.sessions import (
    clear_session_cookie,
    compute_expiry,
    new_session_id,
    set_session_cookie,
)
from app.services.user_service import to_user_out


class AuthService:
    def __init__(self, db: DbSession):
        self.db = db
        self.users = UserRepository(db)
        self.sessions = SessionRepository(db)

    def microsoft_login(self, payload: MicrosoftLoginIn, response: Response) -> UserOut:
        try:
            claims = validate_microsoft_id_token(payload.id_token)
        except MicrosoftIdentityError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        tenant_id = str(claims["tid"]).strip()
        microsoft_oid = str(claims["oid"]).strip()
        email_claim = claims.get("email") or claims.get("preferred_username")
        email = email_claim.lower().strip() if isinstance(email_claim, str) else None

        user = self.users.get_by_tenant_oid(tenant_id, microsoft_oid)
        if user is None:
            admin_oids = parse_admin_oids()
            if microsoft_oid not in admin_oids:
                raise HTTPException(
                    status_code=403,
                    detail="User is not registered. Ask admin to create your account.",
                )
            user = self.users.create(
                User(
                    tenant_id=tenant_id,
                    microsoft_oid=microsoft_oid,
                    email=email,
                    role=AppRole.admin.value,
                    auth_provider="microsoft",
                    is_active=True,
                )
            )

        if not user.is_active:
            raise HTTPException(status_code=403, detail="User is inactive")

        session_id = new_session_id()
        self.sessions.create(Session(id=session_id, user_id=user.id, expires_at=compute_expiry()))
        self.db.commit()
        set_session_cookie(response, session_id)
        return to_user_out(user)

    def logout(self, response: Response, session_id: str | None) -> dict[str, bool]:
        if session_id:
            session = self.sessions.get_by_id(session_id)
            if session:
                self.sessions.delete(session)
                self.db.commit()

        clear_session_cookie(response)
        return {"ok": True}
