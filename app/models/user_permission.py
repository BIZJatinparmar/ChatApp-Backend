from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, uuid_str


class UserPermission(Base):
    __tablename__ = "user_permissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    permission_code: Mapped[str] = mapped_column(String(100), nullable=False)

    user = relationship("User", back_populates="permissions")

    __table_args__ = (
        UniqueConstraint("user_id", "permission_code", name="uq_user_permission"),
        Index("ix_user_permissions_user_code", "user_id", "permission_code"),
    )
