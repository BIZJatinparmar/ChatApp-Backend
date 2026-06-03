from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import get_db
from deps.auth import require_permissions
from models.conversation import Conversation
from models.message import Message
from models.user import User
from schemas.models import MessageCreateRequest
from respositories.message_repository import MessageRepository

router = APIRouter(prefix="/message", tags=["message"])


def get_message_repository(db: Session = Depends(get_db)) -> MessageRepository:
    return MessageRepository(db)


@router.post("/messages")
async def insert_message(
        message: MessageCreateRequest,
        db: Session = Depends(get_db),
        user: User = Depends(require_permissions("message:write", "conversation:read")),
        message_repository: MessageRepository = Depends(get_message_repository)):

    try:
        conversation = db.query(Conversation).where(
            Conversation.id == message.conversationId
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.owner_id != user.id:
            raise HTTPException(status_code=403, detail="Conversation is not owned by user")

        message_db = Message(
            id=message.id,
            content=message.content,
            conversation_id=message.conversationId,
            role=message.role
        )
        new_message = message_repository.create_message(message_db)
        db.commit()
        return {
            'message': new_message
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
