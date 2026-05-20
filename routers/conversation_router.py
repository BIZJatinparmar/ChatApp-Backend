
import secrets

from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session
from db import get_db
from models.conversation import Conversation
from pydantic import BaseModel
from respositories.conversation_repository import ConversationRepository
from schemas.models import ConversationCreateRequest


NANO_ID_ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NANO_ID_LENGTH = 21


def generate_nano_id() -> str:
    return "".join(secrets.choice(NANO_ID_ALPHABET) for _ in range(NANO_ID_LENGTH))


class ConversationResponse(BaseModel):
    id: int
    owner_id: int
    title: str

    model_config = {"from_attributes": True}


router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("/")
def conversations(
    db: Session = Depends(get_db)
):
    conversation_repository = ConversationRepository(
        db)
    conversation_data = conversation_repository.get_all_conversations()
    transformed = [{"id": data.id, "title": data.title, "ownerId": data.owner_id,
                    "createdAt": data.created_at, "updatedAt": data.updated_at, "inputTokens": data.input_tokens, "outputTokens": data.output_tokens, "totalTokens": data.total_tokens} for data in conversation_data]

    return {
        'conversations': transformed
    }


@router.get("/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str, db: Session = Depends(get_db)):
    conversation_repository = ConversationRepository(db)
    message_data = conversation_repository.get_conversation_messages(
        conversation_id)
    transformed = [{"id": data.id, "content": data.content, "role": data.role,
                    "conversationId": data.conversation_id, "createdAt": data.created_at, "modelId": data.model_id, "inputTokens": data.input_tokens, "outputTokens": data.output_tokens, "totalTokens": data.total_tokens} for data in message_data]
    return {
        'messages': transformed
    }


@router.post("/")
async def create_conversation(
    body: ConversationCreateRequest,
    # user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation_repository = ConversationRepository(db)
    conversation_id = body.id or generate_nano_id()
    convo = Conversation(id=conversation_id, owner_id="1", title="NewChat")
    conversation_repository.create_conversation(convo)
    return convo
