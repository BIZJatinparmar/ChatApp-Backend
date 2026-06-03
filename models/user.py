from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Base, uuid_str
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    DateTime,
    Integer,
    Boolean,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .conversation import Conversation
    from .user_permission import UserPermission
    from .budget_request import BudgetRequest


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=uuid_str)
    email: Mapped[Optional[str]] = mapped_column(
        String(320), unique=True, index=True, nullable=True)
    # Legacy local-auth column kept for backward compatibility.
    password_hash: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, default="")
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    microsoft_oid: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="user", server_default="user")
    auth_provider: Mapped[str] = mapped_column(
        String(20), nullable=False, default="microsoft", server_default="microsoft")
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1")
    input_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    total_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    token_budget: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    permissions: Mapped[list["UserPermission"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    budget_requests: Mapped[list["BudgetRequest"]] = relationship(
        "BudgetRequest",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="BudgetRequest.user_id",
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "microsoft_oid", name="uq_users_tenant_oid"),
    )
