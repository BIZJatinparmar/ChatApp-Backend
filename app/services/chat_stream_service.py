from __future__ import annotations

import json
import os
import ssl
from typing import Literal, NotRequired, TypedDict

import certifi
import httpx
import tiktoken
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, UsageMetadata
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_vector_db
from app.models.document_model import Document as DbDocument
from app.models.document_model import DocumentStatus
from app.models.message import Message
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.chat import ChatMode, ModelList, Role, StreamMessageRequest
from app.services.rag_service import RagService, format_context

load_dotenv()

ZSCALER_CERT_PATH = os.environ.get(
    "ZSCALER_CERT_PATH",
    "C:\\Users\\jatin.parmar\\Documents\\zscaler.crt",
)


def build_ssl_context() -> ssl.SSLContext:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.load_verify_locations(cafile=ZSCALER_CERT_PATH)
    return ssl_context


custom_ssl_context = build_ssl_context()
custom_client = httpx.Client(verify=custom_ssl_context, trust_env=True)
custom_async_client = httpx.AsyncClient(
    verify=custom_ssl_context,
    trust_env=True,
)


Route = Literal["general_chat", "document_question", "needs_clarification"]


class UsageAccumulator(TypedDict):
    router_input_tokens: int
    router_output_tokens: int
    answer_input_tokens: int
    answer_output_tokens: int


class MessageRouterStructure(BaseModel):
    """Route and optional rewrite for the latest user message."""

    route: Route = Field(
        description="Whether the latest user message should use general chat, document retrieval, or clarification."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the selected route."
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Standalone retrieval query when route is document_question; null otherwise.",
    )
    reasoning: str = Field(description="Short reason for the route decision.")


class AgentState(TypedDict):
    payload: StreamMessageRequest
    user: User
    model_name: str
    history: list[HumanMessage | AIMessage]
    chat_mode: ChatMode
    route: Route | Literal["needs_router"]
    route_reason: str
    route_confidence: Literal["high", "medium", "low"]
    retrieval_query: str
    rag_docs: list[Document]
    context: list[dict]
    system_message: SystemMessage
    answer_kind: Literal["general", "documents", "clarification"]
    citations: list[dict]
    usage: UsageAccumulator
    status_events: list[str]
    router_used: bool
    retrieval_used: bool
    no_document_context: bool
    available_document_count: int
    response_override: NotRequired[str]


def to_ndjson(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def get_llm(model_name: str, temperature: float = 0) -> AzureAIOpenAIApiChatModel:
    return AzureAIOpenAIApiChatModel(
        model=model_name,
        temperature=temperature,
        endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        credential=os.environ["AZURE_API_KEY"],
        http_client=custom_client,
        http_async_client=custom_async_client,
        http_socket_options=(),
        use_responses_api=False,
    )


class ChatStreamService:
    models_list = {}

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

    def __init__(self, db: Session):
        self.db = db
        self.message_repository = MessageRepository(self.db)
        self.conversation_repository = ConversationRepository(self.db)
        self.rag_service = RagService(get_vector_db())
        self.models_list = {model.value: get_llm(
            model.value) for model in ModelList}
        self.graph = self._build_graph()

    @staticmethod
    def _empty_usage() -> UsageAccumulator:
        return {
            "router_input_tokens": 0,
            "router_output_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
        }

    @staticmethod
    def _parse_usage(usage: UsageMetadata | None) -> tuple[int, int, int]:
        if not usage:
            return 0, 0, 0
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        return input_tokens, output_tokens, total_tokens

    @staticmethod
    def _estimate_token_count(content: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name="gpt-5")
        return len(encoding.encode(content))

    @staticmethod
    def _message_text(messages: list[SystemMessage | HumanMessage | AIMessage]) -> str:
        return "\n".join(
            str(message.content)
            for message in messages
            if isinstance(message.content, str)
        )

    def _count_tokens(self, model: AzureAIOpenAIApiChatModel, content: str) -> int:
        try:
            return int(model.get_num_tokens(content))
        except Exception:
            return self._estimate_token_count(content)

    def _available_document_count(self, user: User) -> int:
        return (
            self.db.query(DbDocument)
            .filter(
                DbDocument.owner_id == user.id,
                DbDocument.status == DocumentStatus.READY.value,
            )
            .count()
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("fast_route", self._fast_route)
        graph.add_node("router_and_rewrite", self._router_and_rewrite)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("build_answer_prompt", self._build_answer_prompt)

        graph.set_entry_point("fast_route")
        graph.add_conditional_edges(
            "fast_route",
            self._after_route,
            {
                "router": "router_and_rewrite",
                "retrieve": "retrieve_documents",
                "answer": "build_answer_prompt",
            },
        )
        graph.add_conditional_edges(
            "router_and_rewrite",
            self._after_route,
            {
                "router": "build_answer_prompt",
                "retrieve": "retrieve_documents",
                "answer": "build_answer_prompt",
            },
        )
        graph.add_edge("retrieve_documents", "build_answer_prompt")
        graph.add_edge("build_answer_prompt", END)
        return graph.compile()

    def _fast_route(self, state: AgentState) -> dict:
        chat_mode = state["chat_mode"]
        user_content = state["payload"].user_content
        lowered = user_content.lower()
        available_document_count = self._available_document_count(
            state["user"])

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

    def _after_route(self, state: AgentState) -> str:
        if state["route"] == "needs_router":
            return "router"
        if state["route"] == "document_question":
            return "retrieve"
        return "answer"

    def _router_and_rewrite(self, state: AgentState) -> dict:
        router_model = ModelList.gpt_5_4_nano_1.value
        router_llm = get_llm(router_model, temperature=0)
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
        structured_response = structured_router.invoke(
            [HumanMessage(content=prompt)])
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
        router_input_tokens, router_output_tokens, _ = self._parse_usage(
            raw_usage)
        usage["router_input_tokens"] += (
            router_input_tokens or self._count_tokens(router_llm, prompt)
        )
        usage["router_output_tokens"] += (
            router_output_tokens or self._count_tokens(
                router_llm, response_text)
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

    def _retrieve_documents(self, state: AgentState) -> dict:
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

    def _build_answer_prompt(self, state: AgentState) -> dict:
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
            prompt = self._document_system_prompt(state["context"])
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

    def _document_system_prompt(self, context: list[dict]) -> str:
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

    def _initial_state(
        self,
        payload: StreamMessageRequest,
        user: User,
        history: list[HumanMessage | AIMessage],
        model_name: str,
    ) -> AgentState:
        return {
            "payload": payload,
            "user": user,
            "model_name": model_name,
            "history": history,
            "chat_mode": payload.chat_mode,
            "route": "needs_router",
            "route_reason": "",
            "route_confidence": "low",
            "retrieval_query": payload.user_content,
            "rag_docs": [],
            "context": [],
            "system_message": SystemMessage("You are a helpful AI assistant."),
            "answer_kind": "general",
            "citations": [],
            "usage": self._empty_usage(),
            "status_events": [],
            "router_used": False,
            "retrieval_used": False,
            "no_document_context": False,
            "available_document_count": 0,
        }

    async def stream_messages(self, payload: StreamMessageRequest, user: User):
        parts: list[str] = []

        model_enum = (
            ModelList(payload.model_id)
            if payload.model_id in ModelList._value2member_map_
            else None
        )
        model = (
            self.models_list.get(ModelList.gpt_5_nano.value)
            if model_enum is None
            else self.models_list.get(model_enum.value)
        )
        selected_model_name = (
            ModelList.gpt_5_nano.value if model_enum is None else model_enum.value
        )

        if not model:
            raise ValueError(f"Model {payload.model_id} not found")

        user_input_tokens = self._count_tokens(model, payload.user_content)
        user_message = Message(
            id=payload.message_id,
            role=Role.user,
            content=payload.user_content,
            conversation_id=payload.conversation_id,
            model_id=payload.model_id,
            input_tokens=user_input_tokens,
            total_tokens=user_input_tokens,
        )
        self.message_repository.create_message(user_message)

        existing_messages = self.message_repository.get_all_messages(
            payload.conversation_id
        )
        if len(existing_messages) == 1:
            self.conversation_repository.update_conversation_title(
                payload.conversation_id, user_message.content[:10], user.id
            )

        history = [
            HumanMessage(content=message.content)
            if message.role == Role.user
            else AIMessage(content=message.content)
            for message in existing_messages
        ]

        graph_state = self.graph.invoke(
            self._initial_state(payload, user, history, selected_model_name)
        )

        seen_statuses: set[str] = set()
        for status in graph_state["status_events"]:
            if status in seen_statuses:
                continue
            seen_statuses.add(status)
            yield to_ndjson({"type": "status", "status": status})

        for citation in graph_state["citations"]:
            yield to_ndjson({"type": "citation", "citation": citation})

        usage = dict(graph_state["usage"])
        total_answer_input_tokens = 0
        total_answer_output_tokens = 0

        try:
            if "response_override" in graph_state:
                full_response = graph_state["response_override"].strip()
                usage["answer_input_tokens"] += 0
                usage["answer_output_tokens"] += self._count_tokens(
                    model, full_response)
                parts.append(full_response)
                yield to_ndjson({"type": "token", "content": full_response})
            else:
                langchain_messages: list[SystemMessage | HumanMessage | AIMessage] = [
                    graph_state["system_message"],
                    *history,
                ]
                langchain_content = self._message_text(langchain_messages)

                async for chunk in model.astream(langchain_messages):
                    usage_metadata = (
                        chunk.usage_metadata
                        if hasattr(chunk, "usage_metadata")
                        else None
                    )

                    if usage_metadata:
                        answer_input_tokens, answer_output_tokens, _ = self._parse_usage(
                            usage_metadata
                        )
                        total_answer_input_tokens = answer_input_tokens
                        total_answer_output_tokens += answer_output_tokens
                    else:
                        content_for_count = getattr(chunk, "text", "")
                        total_answer_input_tokens = self._estimate_token_count(
                            langchain_content
                        )
                        total_answer_output_tokens += self._estimate_token_count(
                            content_for_count
                        )

                    content = getattr(chunk, "text", "")
                    if not content:
                        continue

                    parts.append(content)
                    yield to_ndjson({"type": "token", "content": content})

                usage["answer_input_tokens"] += total_answer_input_tokens
                usage["answer_output_tokens"] += total_answer_output_tokens

            full_response = "".join(parts).strip()
            assistant_input_tokens = (
                usage["router_input_tokens"] + usage["answer_input_tokens"]
            )
            assistant_output_tokens = (
                usage["router_output_tokens"] + usage["answer_output_tokens"]
            )
            assistant_total_tokens = assistant_input_tokens + assistant_output_tokens

            assistant_message = Message(
                content=full_response,
                role=Role.assistant,
                conversation_id=payload.conversation_id,
                model_id=payload.model_id,
                input_tokens=assistant_input_tokens,
                output_tokens=assistant_output_tokens,
                total_tokens=assistant_total_tokens,
                payload_json={
                    **usage,
                    "route": graph_state["route"],
                    "route_reason": graph_state["route_reason"],
                    "route_confidence": graph_state["route_confidence"],
                    "router_used": graph_state["router_used"],
                    "retrieval_used": graph_state["retrieval_used"],
                    "retrieval_query": graph_state["retrieval_query"],
                    "answer_kind": graph_state["answer_kind"],
                    "citation_count": len(graph_state["citations"]),
                },
            )

            self.message_repository.create_message(assistant_message)
            self.db.commit()

            yield to_ndjson({"type": "done"})

        except Exception as e:
            self.db.rollback()
            yield to_ndjson({"type": "error", "message": str(e)})
