from sqlalchemy.orm import Session as DbSession

from app.models.session import Session


class SessionRepository:
    def __init__(self, db: DbSession):
        self.db = db

    def get_by_id(self, session_id: str) -> Session | None:
        return self.db.get(Session, session_id)

    def create(self, session: Session) -> Session:
        self.db.add(session)
        self.db.flush()
        return session

    def delete(self, session: Session) -> None:
        self.db.delete(session)
        self.db.flush()
