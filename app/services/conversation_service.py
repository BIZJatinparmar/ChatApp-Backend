import secrets

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.usage_repository import UsageRepository
from app.schemas.conversation import ConversationCreateRequest, ConversationListResponse
from app.schemas.message import ConversationMessagesResponse


NANO_ID_ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NANO_ID_LENGTH = 21


def generate_nano_id() -> str:
    return "".join(secrets.choice(NANO_ID_ALPHABET) for _ in range(NANO_ID_LENGTH))


class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.conversations = ConversationRepository(db)
        self.usage = UsageRepository(db)

    def list_for_user(self, user: User) -> ConversationListResponse:
        conversation_data = self.conversations.get_all_conversations(user.id)
        usage_by_conversation = self.usage.usage_for_conversations(
            [conversation.id for conversation in conversation_data]
        )
        return ConversationListResponse(
            conversations=[
                {
                    "id": data.id,
                    "title": data.title,
                    "ownerId": data.owner_id,
                    "createdAt": data.created_at,
                    "updatedAt": data.updated_at,
                    "inputTokens": usage_by_conversation[data.id]["input_tokens"],
                    "outputTokens": usage_by_conversation[data.id]["output_tokens"],
                    "totalTokens": usage_by_conversation[data.id]["total_tokens"],
                }
                for data in conversation_data
            ]
        )

    def list_messages(self, conversation_id: str, user: User) -> ConversationMessagesResponse:
        conversation = self.conversations.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.owner_id != user.id:
            raise HTTPException(status_code=403, detail="Conversation is not owned by user")

        message_data = self.conversations.get_conversation_messages(conversation_id)
        return ConversationMessagesResponse(
            messages=[
                {
                    "id": data.id,
                    "content": data.content,
                    "role": data.role,
                    "conversationId": data.conversation_id,
                    "createdAt": data.created_at,
                    "modelId": data.model_id,
                    "inputTokens": data.input_tokens,
                    "outputTokens": data.output_tokens,
                    "totalTokens": data.total_tokens,
                }
                for data in message_data
            ]
        )

    def create_conversation(self, body: ConversationCreateRequest, user: User) -> Conversation:
        conversation_id = body.id or generate_nano_id()
        conversation = Conversation(id=conversation_id, owner_id=user.id, title="NewChat")
        self.conversations.create_conversation(conversation)
        return conversation

    def ensure_owned_conversation(self, conversation_id: str, user: User) -> Conversation:
        conversation = self.conversations.get_by_id(conversation_id)
        if conversation and conversation.owner_id != user.id:
            raise HTTPException(status_code=403, detail="Conversation is not owned by user")
        if conversation is None:
            conversation = self.conversations.create_conversation(
                Conversation(id=conversation_id, owner_id=user.id, title="NewChat")
            )
        return conversation
