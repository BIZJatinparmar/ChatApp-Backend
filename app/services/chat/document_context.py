from time import perf_counter

from sqlalchemy.orm import Session

from app.core.database import get_vector_db
from app.models.document_model import Document as DbDocument
from app.models.document_model import DocumentStatus
from app.models.user import User
from app.services.chat.types import AgentState
from app.services.rag_service import RagService, format_context


class ChatDocumentContext:
    def __init__(self, db: Session, rag_service: RagService | None = None):
        self.db = db
        self.rag_service = rag_service or RagService(get_vector_db())

    def available_document_count(self, user: User) -> int:
        return (
            self.db.query(DbDocument)
            .filter(
                DbDocument.owner_id == user.id,
                DbDocument.status == DocumentStatus.READY.value,
            )
            .count()
        )

    def retrieve_documents(self, state: AgentState) -> dict:
        if state["available_document_count"] == 0:
            return {
                "rag_docs": [],
                "context": [],
                "retrieval_used": False,
                "no_document_context": True,
                "fallback_reason": "no_ready_documents",
            }

        started_at = perf_counter()
        result = self.rag_service.get_ranked_context_result(
            state["retrieval_query"],
            state["user"],
            initial_k=state["initial_k"] or 20,
            final_k=state["final_k"] or 8,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        no_document_context = result.retrieval_confidence == "low"
        docs = [] if no_document_context else result.selected_docs
        return {
            "rag_docs": docs,
            "context": format_context(docs),
            "retrieval_used": True,
            "no_document_context": no_document_context,
            "retrieval_confidence": result.retrieval_confidence,
            "retrieved_chunk_count": len(result.retrieved_docs),
            "selected_chunk_count": len(result.selected_docs),
            "retrieved_document_ids": result.retrieved_document_ids,
            "selected_context": result.selected_context,
            "fallback_reason": result.fallback_reason,
            "retrieval_latency_ms": latency_ms,
        }
