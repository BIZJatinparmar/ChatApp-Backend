
from __future__ import annotations

from .Base import Base, uuid_str

from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .Conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=uuid_str)

    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[str] = mapped_column(
        String(20), index=True)  # user/assistant/system/tool
    content: Mapped[str] = mapped_column(Text, nullable=False)

    payload_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_created",
              "conversation_id", "created_at"),
    )
