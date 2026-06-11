import os

from fastapi import HTTPException
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.chat import StreamMessageRequest
from app.services.chat_stream_service import ChatStreamService
from app.services.conversation_service import ConversationService
from app.services.rag_service import format_context


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.conversations = ConversationService(db)
        self.stream_service = ChatStreamService(db)

    def get_docs(self, payload: StreamMessageRequest) -> list[dict]:
        embeddings = OpenAIEmbeddings()
        faiss_path = os.path.join(os.getcwd(), "../faiss_indices")
        vector_db = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "score_threshold": 0.7},
        )
        docs = retriever.invoke(payload.user_content)
        return format_context(docs)

    def stream_messages(self, payload: StreamMessageRequest, user: User):
        self.enforce_token_budget(user)
        self.conversations.ensure_owned_conversation(payload.conversation_id, user)
        return self.stream_service.stream_messages(payload, user)

    @staticmethod
    def enforce_token_budget(user: User) -> None:
        budget_is_enforced = user.role != "admin" or user.token_budget > 0
        if budget_is_enforced and user.total_tokens >= user.token_budget:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "TOKEN_BUDGET_EXCEEDED",
                    "message": "Token budget exceeded. Request additional budget from your admin.",
                    "total_tokens": user.total_tokens,
                    "token_budget": user.token_budget,
                },
            )
