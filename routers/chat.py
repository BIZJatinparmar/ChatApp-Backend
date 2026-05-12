from fastapi import Depends
from fastapi import APIRouter
from db import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from schemas.models import StreamMessageRequest
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os
from services.chat_stream_service import ChatStreamService
from services.rag_service import format_context


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/docs")
async def get_docs(payload: StreamMessageRequest):
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
):
    chat_stream_service = ChatStreamService(db)

    return StreamingResponse(
        chat_stream_service.stream_messages(payload),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
