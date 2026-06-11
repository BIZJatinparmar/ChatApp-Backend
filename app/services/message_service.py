from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.message import MessageCreateRequest


class MessageService:
    def __init__(self, db: Session):
        self.db = db
        self.conversations = ConversationRepository(db)
        self.messages = MessageRepository(db)

    def create_message(self, payload: MessageCreateRequest, user: User) -> dict[str, Message]:
        try:
            conversation = self.conversations.get_by_id(payload.conversationId)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if conversation.owner_id != user.id:
                raise HTTPException(status_code=403, detail="Conversation is not owned by user")

            message = self.messages.create_message(
                Message(
                    id=payload.id,
                    content=payload.content,
                    conversation_id=payload.conversationId,
                    role=payload.role,
                )
            )
            self.db.commit()
            return {"message": message}
        except HTTPException:
            self.db.rollback()
            raise
        except Exception as exc:
            self.db.rollback()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
