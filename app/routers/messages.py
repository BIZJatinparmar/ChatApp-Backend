from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.dependencies.auth import require_permissions
from app.models.user import User
from app.schemas.message import MessageCreateRequest
from app.services.message_service import MessageService

router = APIRouter(prefix="/message", tags=["message"])


@router.post("/messages")
async def insert_message(
        message: MessageCreateRequest,
        db: Session = Depends(get_db),
        user: User = Depends(require_permissions("message:write", "conversation:read"))):
    return MessageService(db).create_message(message, user)
