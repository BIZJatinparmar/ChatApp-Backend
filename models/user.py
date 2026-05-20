from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Base, uuid_str
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    DateTime,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .conversation import Conversation


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=uuid_str)
    email: Mapped[Optional[str]] = mapped_column(
        String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0")
    total_tokens: Mapped[int] = mapped_column(
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
