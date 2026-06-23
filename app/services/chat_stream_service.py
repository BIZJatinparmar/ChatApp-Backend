from __future__ import annotations

import re

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.rag_retrieval_event import RagRetrievalEvent
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.chat import ModelList, Role, StreamMessageRequest
from app.services.chat.document_context import ChatDocumentContext
from app.services.chat.llm_client import build_ssl_context, get_llm
from app.services.chat.ndjson import to_ndjson
from app.services.chat.prompt_builder import ChatPromptBuilder
from app.services.chat.route_planner import ChatRoutePlanner
from app.services.chat.tokens import (
    count_tokens,
    empty_usage,
    estimate_token_count,
    message_text,
    parse_usage,
)
from app.services.chat.types import AgentState


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\](?:\([^)]*\))?")


def extract_citation_indexes(content: str) -> set[int]:
    return {
        int(match)
        for match in _CITATION_MARKER_RE.findall(content)
        if int(match) > 0
    }


def filter_citations_for_answer(answer: str, citations: list[dict]) -> list[dict]:
    used_indexes = extract_citation_indexes(answer)
    if not used_indexes:
        return []

    filtered: list[dict] = []
    seen_indexes: set[int] = set()
    for citation in citations:
        index = citation.get("index")
        if not isinstance(index, int):
            continue
        if index not in used_indexes or index in seen_indexes:
            continue

        filtered.append(citation)
        seen_indexes.add(index)

    return filtered


class ChatStreamService:
    models_list = {}

    def __init__(self, db: Session):
        self.db = db
        self.message_repository = MessageRepository(self.db)
        self.conversation_repository = ConversationRepository(self.db)
        self.document_context = ChatDocumentContext(self.db)
        self.route_planner = ChatRoutePlanner(
            self.document_context.available_document_count
        )
        self.prompt_builder = ChatPromptBuilder()
        self.models_list = {model.value: get_llm(model.value) for model in ModelList}
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("fast_route", self.route_planner.fast_route)
        graph.add_node("router_and_rewrite", self.route_planner.router_and_rewrite)
        graph.add_node("plan_retrieval", self.route_planner.plan_retrieval)
        graph.add_node("retrieve_documents", self.document_context.retrieve_documents)
        graph.add_node("build_answer_prompt", self.prompt_builder.build_answer_prompt)

        graph.set_entry_point("fast_route")
        graph.add_conditional_edges(
            "fast_route",
            self.route_planner.after_route,
            {
                "router": "router_and_rewrite",
                "plan": "plan_retrieval",
                "answer": "build_answer_prompt",
            },
        )
        graph.add_conditional_edges(
            "router_and_rewrite",
            self.route_planner.after_route,
            {
                "router": "build_answer_prompt",
                "plan": "plan_retrieval",
                "answer": "build_answer_prompt",
            },
        )
        graph.add_conditional_edges(
            "plan_retrieval",
            self.route_planner.after_plan,
            {
                "retrieve": "retrieve_documents",
                "answer": "build_answer_prompt",
            },
        )
        graph.add_edge("retrieve_documents", "build_answer_prompt")
        graph.add_edge("build_answer_prompt", END)
        return graph.compile()

    def _initial_state(
        self,
        payload: StreamMessageRequest,
        user: User,
        history: list[HumanMessage | AIMessage],
        model_name: str,
    ) -> AgentState:
        state: AgentState = {
            "payload": payload,
            "user": user,
            "model_name": model_name,
            "history": history,
            "chat_mode": payload.chat_mode,
            "route": "needs_router",
            "route_reason": "",
            "route_confidence": "low",
            "original_query": payload.user_content,
            "retrieval_query": payload.user_content,
            "rewrite_used": False,
            "rewrite_source": "none",
            "rewrite_confidence": "low",
            "initial_k": 0,
            "final_k": 0,
            "retrieval_confidence": "low",
            "retrieved_chunk_count": 0,
            "selected_chunk_count": 0,
            "retrieved_document_ids": [],
            "selected_context": [],
            "fallback_reason": None,
            "retrieval_latency_ms": 0,
            "rag_docs": [],
            "context": [],
            "system_message": SystemMessage("You are a helpful AI assistant."),
            "answer_kind": "general",
            "citations": [],
            "usage": empty_usage(),
            "status_events": [],
            "router_used": False,
            "retrieval_used": False,
            "no_document_context": False,
            "available_document_count": 0,
        }
        return state

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

        user_input_tokens = count_tokens(model, payload.user_content)
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

        usage = dict(graph_state["usage"])
        total_answer_input_tokens = 0
        total_answer_output_tokens = 0

        try:
            if "response_override" in graph_state:
                full_response = graph_state["response_override"].strip()
                usage["answer_input_tokens"] += 0
                usage["answer_output_tokens"] += count_tokens(model, full_response)
                parts.append(full_response)
                yield to_ndjson({"type": "token", "content": full_response})
            else:
                langchain_messages: list[SystemMessage | HumanMessage | AIMessage] = [
                    graph_state["system_message"],
                    *history,
                ]
                langchain_content = message_text(langchain_messages)

                async for chunk in model.astream(langchain_messages):
                    usage_metadata = (
                        chunk.usage_metadata
                        if hasattr(chunk, "usage_metadata")
                        else None
                    )

                    if usage_metadata:
                        answer_input_tokens, answer_output_tokens, _ = parse_usage(
                            usage_metadata
                        )
                        total_answer_input_tokens = answer_input_tokens
                        total_answer_output_tokens += answer_output_tokens
                    else:
                        content_for_count = getattr(chunk, "text", "")
                        total_answer_input_tokens = estimate_token_count(
                            langchain_content
                        )
                        total_answer_output_tokens += estimate_token_count(
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
            final_citations = filter_citations_for_answer(
                full_response,
                graph_state["citations"],
            )
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
                    "original_query": graph_state["original_query"],
                    "retrieval_query": graph_state["retrieval_query"],
                    "rewrite_used": graph_state["rewrite_used"],
                    "rewrite_source": graph_state["rewrite_source"],
                    "rewrite_confidence": graph_state["rewrite_confidence"],
                    "retrieval_confidence": graph_state["retrieval_confidence"],
                    "fallback_reason": graph_state["fallback_reason"],
                    "answer_kind": graph_state["answer_kind"],
                    "citations": final_citations,
                    "citation_count": len(final_citations),
                },
            )

            self.message_repository.create_message(assistant_message)
            self._persist_rag_retrieval_event(graph_state, assistant_message.id)
            self.db.commit()

            for citation in final_citations:
                yield to_ndjson({"type": "citation", "citation": citation})

            yield to_ndjson({"type": "done"})

        except Exception as e:
            self.db.rollback()
            yield to_ndjson({"type": "error", "message": str(e)})

    def _persist_rag_retrieval_event(self, graph_state: AgentState, assistant_message_id: str) -> None:
        if not graph_state["retrieval_used"]:
            return

        self.db.add(
            RagRetrievalEvent(
                message_id=assistant_message_id,
                conversation_id=graph_state["payload"].conversation_id,
                user_id=graph_state["user"].id,
                chat_mode=graph_state["chat_mode"].value,
                route=graph_state["route"],
                route_confidence=graph_state["route_confidence"],
                original_query=graph_state["original_query"],
                rewritten_query=graph_state["retrieval_query"],
                rewrite_used=graph_state["rewrite_used"],
                rewrite_source=graph_state["rewrite_source"],
                rewrite_confidence=graph_state["rewrite_confidence"],
                initial_k=graph_state["initial_k"],
                final_k=graph_state["final_k"],
                retrieval_confidence=graph_state["retrieval_confidence"],
                retrieved_chunk_count=graph_state["retrieved_chunk_count"],
                selected_chunk_count=graph_state["selected_chunk_count"],
                retrieved_document_ids=graph_state["retrieved_document_ids"],
                selected_context=graph_state["selected_context"],
                fallback_reason=graph_state["fallback_reason"],
                latency_ms=graph_state["retrieval_latency_ms"],
            )
        )
