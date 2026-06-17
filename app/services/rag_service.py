from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVectorStore

from app.models.user import User


def format_context(docs: list[Document]) -> list[dict]:
    lines = []
    for doc in docs:
        lines.append(
            {
                "document_id": doc.metadata.get("document_id"),
                "src": doc.metadata.get("filename")
                or doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
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
