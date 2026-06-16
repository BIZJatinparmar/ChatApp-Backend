import enum

from datetime import datetime

from .base import Base, uuid_str
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, ForeignKey, Integer, String, func


class FileType(str, enum.Enum):
    PDF = "pdf"
    TXT = "txt"


class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "document"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=uuid_str)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    filepath: Mapped[str | None] = mapped_column(String(500), nullable=True)
    storage_path: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(100))
    filetype: Mapped[str] = mapped_column(
        String(20), nullable=False, default=FileType.TXT.value)
    size_bytes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DocumentStatus.PROCESSING.value, server_default=DocumentStatus.PROCESSING.value)
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
