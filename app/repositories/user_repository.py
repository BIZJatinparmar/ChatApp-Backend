from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> User | None:
        return self.db.get(User, user_id)

    def get_by_tenant_oid(self, tenant_id: str, microsoft_oid: str) -> User | None:
        return self.db.scalar(
            select(User).where(
                User.tenant_id == tenant_id,
                User.microsoft_oid == microsoft_oid,
            )
        )

    def list_for_tenant(self, tenant_id: str | None) -> list[User]:
        return list(
            self.db.scalars(
                select(User).where(User.tenant_id == tenant_id).order_by(User.created_at.desc())
            ).all()
        )

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.flush()
        self.db.refresh(user)
        return user

    def increment_usage(self, user_id: str, input_tokens: int, output_tokens: int, total_tokens: int):
        user = self.db.query(User).where(User.id == user_id).first()
        if not user:
            return None
        user.input_tokens += int(input_tokens)
        user.output_tokens += int(output_tokens)
        user.total_tokens += int(total_tokens)
        self.db.flush()
        return user
