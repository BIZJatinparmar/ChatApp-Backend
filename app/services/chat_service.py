from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_vector_db
from app.models.user import User
from app.repositories.usage_repository import UsageRepository
from app.schemas.chat import StreamMessageRequest
from app.services.chat_stream_service import ChatStreamService
from app.services.conversation_service import ConversationService
from app.services.rag_service import RagService, format_context


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.conversations = ConversationService(db)
        self.stream_service = ChatStreamService(db)
        self.usage = UsageRepository(db)

    def get_docs(self, payload: StreamMessageRequest, user: User) -> list[dict]:
        docs = RagService(get_vector_db()).get_relevant_documents(
            payload.user_content, user, k=5)
        return format_context(docs)

    def stream_messages(self, payload: StreamMessageRequest, user: User):
        self.enforce_token_budget(user)
        self.conversations.ensure_owned_conversation(payload.conversation_id, user)
        return self.stream_service.stream_messages(payload, user)

    def enforce_token_budget(self, user: User) -> None:
        total_tokens = self.usage.usage_for_user(user.id)["total_tokens"]
        budget_is_enforced = user.role != "admin" or user.token_budget > 0
        if budget_is_enforced and total_tokens >= user.token_budget:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "TOKEN_BUDGET_EXCEEDED",
                    "message": "Token budget exceeded. Request additional budget from your admin.",
                    "total_tokens": total_tokens,
                    "token_budget": user.token_budget,
                },
            )
