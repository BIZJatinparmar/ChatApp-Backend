from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse
from langchain_postgres import PGVectorStore
from sqlalchemy.orm import Session

from app.core.database import get_db, get_vector_db
from app.dependencies.auth import require_permissions
from app.models.document_model import Document
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentOut
from app.services.document_service import DocumentService

router = APIRouter(prefix="/document", tags=["document"])


def serialize_document(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        filetype=document.filetype,
        size_bytes=document.size_bytes,
        status=document.status,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    user: User = Depends(require_permissions("document:manage")),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    documents = DocumentService(db).list_documents(user)
    return DocumentListResponse(
        documents=[serialize_document(document) for document in documents]
    )


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(require_permissions("document:manage")),
    db: Session = Depends(get_db),
    pg_vector_store: PGVectorStore = Depends(get_vector_db),
) -> DocumentOut:
    document = await DocumentService(db, pg_vector_store).upload_document(user, file)
    return serialize_document(document)


@router.get("/{document_id}/preview", response_class=FileResponse)
def preview_document(
    document_id: str,
    user: User = Depends(require_permissions("document:manage")),
    db: Session = Depends(get_db),
) -> FileResponse:
    return DocumentService(db).preview_document(user, document_id)
