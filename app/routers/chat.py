from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.dependencies.auth import require_permissions
from app.models.user import User
from app.schemas.chat import StreamMessageRequest
from app.services.chat_service import ChatService


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/docs")
async def get_docs(
    payload: StreamMessageRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("chat:use")),
):
    return ChatService(db).get_docs(payload, user)


@router.post("/stream")
async def chat_stream(
    payload: StreamMessageRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("chat:use", "conversation:read", "message:write")),
):
    chat_service = ChatService(db)
    return StreamingResponse(
        chat_service.stream_messages(payload, user),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
