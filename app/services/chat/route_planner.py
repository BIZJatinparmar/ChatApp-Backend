from collections.abc import Callable
from typing import Any, Protocol, cast

from langchain.messages import HumanMessage

from app.models.user import User
from app.schemas.chat import ChatMode, ModelList
from app.services.chat.tokens import count_tokens, parse_usage
from app.services.chat.types import AgentState, MessageRouterStructure


class RouterLlm(Protocol):
    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_num_tokens(self, text: str) -> int:
        ...


class ChatRoutePlanner:
    _DOCUMENT_HINTS = {
        "document",
        "doc",
        "pdf",
        "file",
        "uploaded",
        "attachment",
        "contract",
        "agreement",
        "policy",
        "invoice",
        "report",
        "resume",
        "clause",
        "section",
        "page",
        "summarize it",
        "summarise it",
        "this file",
        "the file",
        "the pdf",
        "in the document",
        "according to",
    }
    _GENERAL_HINTS = {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "write",
        "draft",
        "explain",
        "brainstorm",
        "translate",
        "summarize this text",
    }

    def __init__(
        self,
        available_document_count: Callable[[User], int],
        llm_factory: Callable[[str, float], RouterLlm] | None = None,
    ):
        self.available_document_count = available_document_count
        self.llm_factory = llm_factory

    def fast_route(self, state: AgentState) -> dict:
        chat_mode = state["chat_mode"]
        user_content = state["payload"].user_content
        lowered = user_content.lower()
        available_document_count = self.available_document_count(state["user"])

        if chat_mode == ChatMode.general_only:
            return {
                "route": "general_chat",
                "route_reason": "User selected general-only mode.",
                "route_confidence": "high",
                "retrieval_query": user_content,
                "available_document_count": available_document_count,
            }

        if chat_mode == ChatMode.document_only:
            return {
                "route": "document_question",
                "route_reason": "User selected document-only mode.",
                "route_confidence": "high",
                "retrieval_query": user_content,
                "status_events": ["searching_documents"],
                "available_document_count": available_document_count,
            }

        if any(hint in lowered for hint in self._DOCUMENT_HINTS):
            return {
                "route": "document_question",
                "route_reason": "Message contains document-reference keywords.",
                "route_confidence": "high",
                "retrieval_query": user_content,
                "status_events": ["searching_documents"],
                "available_document_count": available_document_count,
            }

        short_greeting = len(user_content.split()) <= 4 and any(
            hint == lowered.strip(" !?.") for hint in self._GENERAL_HINTS
        )
        if short_greeting or available_document_count == 0:
            return {
                "route": "general_chat",
                "route_reason": "Fast path selected general chat.",
                "route_confidence": "high",
                "retrieval_query": user_content,
                "available_document_count": available_document_count,
            }

        return {
            "route": "needs_router",
            "route_reason": "Auto mode question was ambiguous.",
            "route_confidence": "low",
            "retrieval_query": user_content,
            "available_document_count": available_document_count,
        }

    def after_route(self, state: AgentState) -> str:
        if state["route"] == "needs_router":
            return "router"
        if state["route"] == "document_question":
            return "retrieve"
        return "answer"

    def router_and_rewrite(self, state: AgentState) -> dict:
        if self.llm_factory is None:
            from app.services.chat.llm_client import get_llm

            self.llm_factory = cast(Callable[[str, float], RouterLlm], get_llm)

        router_model = ModelList.gpt_5_4_nano_1.value
        router_llm = self.llm_factory(router_model, 0)
        structured_router = router_llm.with_structured_output(
            MessageRouterStructure,
            method="function_calling",
            include_raw=True,
        )
        recent_messages = "\n".join(
            f"{message.type}: {message.content}" for message in state["history"][-6:]
        )
        prompt = f"""Classify whether the latest user message needs uploaded documents.
Return route as one of: general_chat, document_question, needs_clarification.
If the route is document_question, return a standalone rewritten_query for retrieval.
Prefer general_chat unless the user asks about uploaded files or prior document context.

Available document count: {state['available_document_count']}
Recent messages:
{recent_messages}

Latest user message:
{state['payload'].user_content}
"""
        structured_response = structured_router.invoke([HumanMessage(content=prompt)])
        response = structured_response.get("parsed")
        raw_response = structured_response.get("raw")
        parsing_error = structured_response.get("parsing_error")

        if parsing_error or response is None:
            response = MessageRouterStructure(
                route="general_chat",
                confidence="low",
                rewritten_query=None,
                reasoning="Router structured output failed; defaulting to general chat.",
            )

        response_text = response.model_dump_json()

        usage = dict(state["usage"])
        raw_usage = (
            getattr(raw_response, "usage_metadata", None)
            if raw_response is not None
            else None
        )
        router_input_tokens, router_output_tokens, _ = parse_usage(raw_usage)
        usage["router_input_tokens"] += (
            router_input_tokens or count_tokens(router_llm, prompt)
        )
        usage["router_output_tokens"] += (
            router_output_tokens or count_tokens(router_llm, response_text)
        )

        route = response.route
        if route not in {"general_chat", "document_question", "needs_clarification"}:
            route = "general_chat"

        retrieval_query = response.rewritten_query or state["payload"].user_content
        status_events = list(state["status_events"])
        if route == "document_question":
            status_events.append("searching_documents")

        return {
            "route": route,
            "route_reason": response.reasoning,
            "route_confidence": response.confidence,
            "retrieval_query": retrieval_query,
            "usage": usage,
            "status_events": status_events,
            "router_used": True,
        }
