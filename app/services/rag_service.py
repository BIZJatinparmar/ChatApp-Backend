# app/services/rag_service.py

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document


def format_context(docs: list[Document]) -> list[dict]:
    lines = []
    for d in docs:
        data = {
            "src": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "unknown"),
            "text": d.page_content.strip().replace("\n", " ")
        }
        lines.append(data)

    return lines


class RagService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.faiss_path = os.path.abspath(
            os.path.join(os.getcwd(), "../faiss_indices")
        )

    def build_system_message(self, user_content: str) -> SystemMessage:
        if not os.path.exists(self.faiss_path):
            return SystemMessage("You are a helpful AI assistant.")

        vector_db = FAISS.load_local(
            self.faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        docs = retriever.invoke(user_content)
        context = format_context(docs)

        system_message_text = self._build_prompt_from_context(context)

        return SystemMessage(system_message_text)

    def _build_prompt_from_context(self, context: list[dict]) -> str:
        system_message_text = """
You are a helpful AI assistant.
Use the following context to answer the user's question.
If the answer is not in context, answer based on your own knowledge.

Context:
"""

        for c in context:
            system_message_text += (
                f"- Source: {c['src']}, Page: {c['page']}\n"
                f"  Text: {c['text']}\n"
            )

        return system_message_text
