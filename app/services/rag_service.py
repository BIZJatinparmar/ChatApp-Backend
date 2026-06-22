from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVectorStore

from app.models.user import User


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float | None
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    selected_docs: list[Document]
    retrieved_docs: list[ScoredDocument]
    selected_context: list[dict]
    retrieved_document_ids: list[str]
    retrieval_confidence: str
    fallback_reason: str | None


def format_context(docs: list[Document]) -> list[dict]:
    lines = []
    for index, doc in enumerate(docs, start=1):
        lines.append(
            {
                "document_id": doc.metadata.get("document_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "src": doc.metadata.get("filename")
                or doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "start_index": doc.metadata.get("start_index"),
                "rank": doc.metadata.get("retrieval_rank", index),
                "score": doc.metadata.get("retrieval_score"),
                "text": doc.page_content.strip().replace("\n", " "),
            }
        )

    return lines


class RagService:
    def __init__(self, vector_store: PGVectorStore):
        self.vector_store = vector_store

    def get_relevant_documents(self, user_content: str, user: User, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(
            user_content,
            k=k,
            filter={"owner_id": user.id},
        )

    def get_relevant_documents_with_scores(
        self,
        user_content: str,
        user: User,
        k: int = 4,
    ) -> list[ScoredDocument]:
        search_with_score = getattr(
            self.vector_store, "similarity_search_with_score", None)
        print("search_with_score:", search_with_score)
        if callable(search_with_score):
            results = search_with_score(
                user_content,
                k=k,
                filter={"owner_id": user.id},
            )
            return [
                ScoredDocument(
                    document=self._with_retrieval_metadata(
                        doc, rank=index, score=score),
                    score=self._coerce_score(score),
                    rank=index,
                )
                for index, (doc, score) in enumerate(results, start=1)
            ]

        docs = self.get_relevant_documents(user_content, user, k=k)
        return [
            ScoredDocument(
                document=self._with_retrieval_metadata(
                    doc, rank=index, score=None),
                score=None,
                rank=index,
            )
            for index, doc in enumerate(docs, start=1)
        ]

    def get_ranked_context(
        self,
        user_content: str,
        user: User,
        initial_k: int = 20,
        final_k: int = 8,
    ) -> list[Document]:
        docs = self.get_relevant_documents(user_content, user, k=initial_k)
        deduped: list[Document] = []
        seen: set[tuple[str, str, str]] = set()

        for doc in docs:
            key = (
                str(doc.metadata.get("document_id", "")),
                str(doc.metadata.get("page", "")),
                doc.page_content.strip()[:300],
            )
            if key in seen:
                continue

            seen.add(key)
            deduped.append(doc)

            if len(deduped) >= final_k:
                break

        return deduped

    def get_ranked_context_result(
        self,
        user_content: str,
        user: User,
        initial_k: int = 20,
        final_k: int = 8,
    ) -> RetrievalResult:
        retrieved_docs = self.get_relevant_documents_with_scores(
            user_content,
            user,
            k=initial_k,
        )
        print("user_content:", user_content)
        print("Retrieved Docs:", retrieved_docs)

        selected_docs = self._select_diverse_documents(
            retrieved_docs, final_k=final_k)
        selected_context = self._selected_context(selected_docs)
        retrieved_document_ids = self._unique_document_ids(
            item.document for item in retrieved_docs
        )
        confidence = self._confidence(
            query=user_content,
            retrieved_count=len(retrieved_docs),
            selected_count=len(selected_docs),
            selected_scores=[doc.metadata.get(
                "retrieval_score") for doc in selected_docs],
        )
        fallback_reason = None
        if confidence == "low":
            fallback_reason = "low_retrieval_confidence"
        if not selected_docs:
            fallback_reason = "no_retrieval_results"

        return RetrievalResult(
            selected_docs=selected_docs,
            retrieved_docs=retrieved_docs,
            selected_context=selected_context,
            retrieved_document_ids=retrieved_document_ids,
            retrieval_confidence=confidence,
            fallback_reason=fallback_reason,
        )

    def _select_diverse_documents(
        self,
        retrieved_docs: list[ScoredDocument],
        final_k: int,
    ) -> list[Document]:
        selected: list[Document] = []
        seen_chunks: set[str] = set()
        seen_fallbacks: set[tuple[str, str, str]] = set()
        per_page_counts: dict[tuple[str, str], int] = {}

        for item in retrieved_docs:
            doc = item.document
            chunk_id = str(doc.metadata.get("chunk_id") or "")
            fallback_key = (
                str(doc.metadata.get("document_id", "")),
                str(doc.metadata.get("page", "")),
                doc.page_content.strip()[:300],
            )
            if chunk_id and chunk_id in seen_chunks:
                continue
            if not chunk_id and fallback_key in seen_fallbacks:
                continue

            page_key = (
                str(doc.metadata.get("document_id", "")),
                str(doc.metadata.get("page", "")),
            )
            if per_page_counts.get(page_key, 0) >= 2 and len(selected) >= max(2, final_k // 2):
                continue

            if chunk_id:
                seen_chunks.add(chunk_id)
            else:
                seen_fallbacks.add(fallback_key)
            per_page_counts[page_key] = per_page_counts.get(page_key, 0) + 1
            selected.append(doc)

            if len(selected) >= final_k:
                break

        return selected

    def _selected_context(self, docs: list[Document]) -> list[dict]:
        return [
            {
                "document_id": doc.metadata.get("document_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "filename": doc.metadata.get("filename") or doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "quote": doc.page_content.strip().replace("\n", " ")[:300],
                "rank": doc.metadata.get("retrieval_rank"),
                "score": doc.metadata.get("retrieval_score"),
            }
            for doc in docs
        ]

    def _confidence(
        self,
        query: str,
        retrieved_count: int,
        selected_count: int,
        selected_scores: list[Any],
    ) -> str:
        if selected_count == 0:
            return "low"
        numeric_scores = [
            score for score in (self._coerce_score(score) for score in selected_scores)
            if score is not None
        ]
        if numeric_scores:
            best_score = min(numeric_scores)
            if best_score <= 0.4 and selected_count >= 2:
                return "high"
            if best_score <= 0.85:
                return "medium"

        query_word_count = len(query.split())
        if selected_count >= 4 and retrieved_count >= 8 and query_word_count >= 4:
            return "high"
        if selected_count >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _unique_document_ids(docs: list[Document]) -> list[str]:
        ids: list[str] = []
        for doc in docs:
            document_id = doc.metadata.get("document_id")
            if document_id and document_id not in ids:
                ids.append(document_id)
        return ids

    @staticmethod
    def _with_retrieval_metadata(doc: Document, rank: int, score: Any) -> Document:
        metadata = dict(doc.metadata)
        metadata["retrieval_rank"] = rank
        metadata["retrieval_score"] = RagService._coerce_score(score)
        return Document(page_content=doc.page_content, metadata=metadata)

    @staticmethod
    def _coerce_score(score: Any) -> float | None:
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def build_system_message(self, user_content: str, user: User) -> SystemMessage:
        docs = self.get_relevant_documents(user_content, user)
        if not docs:
            return SystemMessage("You are a helpful AI assistant.")

        context = format_context(docs)
        return SystemMessage(self._build_prompt_from_context(context))

    def _build_prompt_from_context(self, context: list[dict]) -> str:
        system_message_text = """
You are a helpful AI assistant.
Use the following context to answer the user's question.
If the answer is not in context, answer based on your own knowledge.

Context:
"""

        for item in context:
            system_message_text += (
                f"- Source: {item['src']}, Page: {item['page']}\n"
                f"  Text: {item['text']}\n"
            )

        return system_message_text
