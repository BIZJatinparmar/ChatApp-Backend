from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from models.message import Message
from schemas.models import MessageCreateRequest
from respositories.message_repository import MessageRepository

router = APIRouter(prefix="/message", tags=["message"])


def get_message_repository(db: Session = Depends(get_db)) -> MessageRepository:
    return MessageRepository(db)


@router.post("/messages")
async def insert_message(
        message: MessageCreateRequest,
        db: Session = Depends(get_db),
        message_repository: MessageRepository = Depends(get_message_repository)):

    try:
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

    except:
        db.rollback()
