from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from sqlalchemy.orm import Session

from db import get_db
from deps.auth import require_permissions
from models.conversation import Conversation
from models.user import User
from respositories.conversation_repository import ConversationRepository
from schemas.models import StreamMessageRequest
from services.chat_stream_service import ChatStreamService
from services.rag_service import format_context


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/docs")
async def get_docs(
    payload: StreamMessageRequest,
    _: User = Depends(require_permissions("chat:use")),
):
    embeddings = OpenAIEmbeddings()
    faiss_path = os.path.join(os.getcwd(), "../faiss_indices")
    vector_db = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={
                                       "k": 5, "score_threshold": 0.7})
    docs = retriever.invoke(payload.user_content)
    return format_context(docs)


@router.post("/stream")
async def chat_stream(
    payload: StreamMessageRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("chat:use", "conversation:read", "message:write")),
):
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

    conversation_repository = ConversationRepository(db)
    conversation = conversation_repository.get_by_id(payload.conversation_id)
    if conversation and conversation.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Conversation is not owned by user")
    if conversation is None:
        conversation_repository.create_conversation(
            Conversation(id=payload.conversation_id, owner_id=user.id, title="NewChat")
        )

    chat_stream_service = ChatStreamService(db)

    return StreamingResponse(
        chat_stream_service.stream_messages(payload, user),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
