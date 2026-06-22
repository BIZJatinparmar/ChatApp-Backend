from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, uuid_str


class RagRetrievalEvent(Base):
    __tablename__ = "rag_retrieval_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    message_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_mode: Mapped[str] = mapped_column(String(30), nullable=False)
    route: Mapped[str] = mapped_column(String(30), nullable=False)
    route_confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewrite_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    rewrite_source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="none", server_default="none"
    )
    rewrite_confidence: Mapped[str] = mapped_column(
        String(20), nullable=False, default="low", server_default="low"
    )
    initial_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    final_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    retrieval_confidence: Mapped[str] = mapped_column(
        String(20), nullable=False, default="low", server_default="low"
    )
    retrieved_chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    selected_chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    retrieved_document_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    selected_context: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_rag_retrieval_events_conversation_created", "conversation_id", "created_at"),
        Index("ix_rag_retrieval_events_user_created", "user_id", "created_at"),
    )
