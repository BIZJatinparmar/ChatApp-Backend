from langchain.messages import SystemMessage

from app.services.chat.types import AgentState


class ChatPromptBuilder:
    def build_answer_prompt(self, state: AgentState) -> dict:
        if state["route"] == "needs_clarification":
            return {
                "answer_kind": "clarification",
                "response_override": "Do you want me to answer from your uploaded documents or generally?",
                "system_message": SystemMessage("Ask one concise clarification question."),
            }

        if state["route"] == "document_question":
            if state["no_document_context"]:
                if state["available_document_count"] == 0:
                    response = "I do not see any ready uploaded documents to search yet. Please upload a document first, or switch to general mode."
                else:
                    response = "I searched your uploaded documents but could not find enough relevant information to answer confidently."

                return {
                    "answer_kind": "documents",
                    "response_override": response,
                    "system_message": SystemMessage(
                        "Explain that the uploaded documents did not contain enough evidence."
                    ),
                    "status_events": state["status_events"] + ["no_document_match"],
                }

            citations = [
                {
                    "documentId": item.get("document_id") or item.get("src"),
                    "fileName": item.get("src", "unknown"),
                    "page": item.get("page", "unknown"),
                    "quote": item.get("text", "")[:300],
                }
                for item in state["context"]
            ]
            prompt = self.document_system_prompt(state["context"])
            return {
                "answer_kind": "documents",
                "system_message": SystemMessage(prompt),
                "citations": citations,
                "status_events": state["status_events"] + ["answering_from_documents"],
            }

        return {
            "answer_kind": "general",
            "system_message": SystemMessage("You are a helpful AI assistant."),
            "status_events": state["status_events"] + ["answering_generally"],
        }

    def document_system_prompt(self, context: list[dict]) -> str:
        prompt = """You are a helpful AI assistant answering from uploaded documents.
Use only the provided document context for factual claims about the documents.
If the answer is not supported by the context, say you could not find it in the uploaded documents.
Cite sources using the file name and page when available.
Do not invent clauses, numbers, dates, names, or obligations.

Document context:
"""
        for index, item in enumerate(context, start=1):
            prompt += (
                f"\n[{index}] Source: {item['src']}, Page: {item['page']}\n"
                f"{item['text']}\n"
            )
        return prompt
