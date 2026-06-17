from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from langchain_core.documents import Document as LangchainDocument
from langchain_postgres import PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.models.document_model import Document, DocumentStatus, FileType
from app.models.user import User


UPLOAD_ROOT = Path(os.environ.get(
    "DOCUMENT_UPLOAD_DIR", "uploaded_documents")).resolve()
SUPPORTED_TYPES = {
    "application/pdf": FileType.PDF.value,
    "text/plain": FileType.TXT.value,
}


class DocumentService:
    def __init__(self, db: Session, vector_store: PGVectorStore | None = None):
        self.db = db
        self.vector_store = vector_store

    def list_documents(self, user: User) -> list[Document]:
        return (
            self.db.query(Document)
            .filter(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
            .all()
        )

    async def upload_document(self, user: User, file: UploadFile) -> Document:
        filetype = SUPPORTED_TYPES.get(file.content_type or "")
        if not filetype:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Only PDF and TXT are allowed.",
            )

        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        document_id = str(uuid4())
        original_filename = Path(file.filename or "document").name
        extension = ".pdf" if filetype == FileType.PDF.value else ".txt"
        storage_dir = UPLOAD_ROOT / user.id
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"{document_id}{extension}"
        storage_path.write_bytes(contents)

        document = Document(
            id=document_id,
            owner_id=user.id,
            filename=original_filename,
            filepath=str(storage_path),
            storage_path=str(storage_path),
            content_type=file.content_type or "application/octet-stream",
            filetype=filetype,
            size_bytes=len(contents),
            status=DocumentStatus.PROCESSING.value,
            chunk_count=0,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        try:
            chunks = self._build_chunks(document, storage_path, contents)
            if not chunks:
                raise ValueError(
                    "No readable text was found in the uploaded file.")

            if self.vector_store is None:
                raise ValueError("Vector store is not configured.")

            self.vector_store.add_documents(chunks)
            document.status = DocumentStatus.READY.value
            document.chunk_count = len(chunks)
            self.db.commit()
            self.db.refresh(document)
            return document
        except Exception as exc:
            document.status = DocumentStatus.FAILED.value
            self.db.commit()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process document: {exc}",
            ) from exc

    def preview_document(self, user: User, document_id: str) -> FileResponse:
        document = self._get_owned_document(user, document_id)
        path = Path(document.storage_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stored document file was not found.",
            )

        return FileResponse(
            path,
            media_type=document.content_type,
            filename=document.filename,
            content_disposition_type="inline",
        )

    def _get_owned_document(self, user: User, document_id: str) -> Document:
        document = (
            self.db.query(Document)
            .filter(Document.id == document_id, Document.owner_id == user.id)
            .first()
        )
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return document

    def _build_chunks(
        self,
        document: Document,
        storage_path: Path,
        contents: bytes,
    ) -> list[LangchainDocument]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        if document.filetype == FileType.PDF.value:
            reader = PdfReader(str(storage_path))
            pages = [
                LangchainDocument(
                    page_content=page.extract_text() or "",
                    metadata=self._metadata(document, page=index + 1),
                )
                for index, page in enumerate(reader.pages)
            ]
            return splitter.split_documents(
                [page for page in pages if page.page_content.strip()]
            )

        text = contents.decode("utf-8", errors="replace")
        source = LangchainDocument(
            page_content=text,
            metadata=self._metadata(document),
        )
        return splitter.split_documents([source])

    @staticmethod
    def _metadata(document: Document, page: int | None = None) -> dict:
        metadata = {
            "source": document.filename,
            "owner_id": document.owner_id,
            "document_id": document.id,
            "filename": document.filename,
            "filetype": document.filetype,
        }
        if page is not None:
            metadata["page"] = str(page)
        return metadata
