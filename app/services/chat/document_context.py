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
            }

        docs = self.rag_service.get_ranked_context(
            state["retrieval_query"],
            state["user"],
            initial_k=20,
            final_k=8,
        )
        return {
            "rag_docs": docs,
            "context": format_context(docs),
            "retrieval_used": True,
            "no_document_context": len(docs) == 0,
        }
