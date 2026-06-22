from typing import Literal, NotRequired, TypedDict

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.models.user import User
from app.schemas.chat import ChatMode, StreamMessageRequest


Route = Literal["general_chat", "document_question", "needs_clarification"]
Confidence = Literal["high", "medium", "low"]
RewriteSource = Literal["llm", "heuristic", "none"]


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
    confidence: Confidence = Field(
        description="Confidence in the selected route."
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Standalone retrieval query when route is document_question; null otherwise.",
    )
    reasoning: str = Field(description="Short reason for the route decision.")


class QueryRewriteStructure(BaseModel):
    """Standalone query for uploaded-document retrieval."""

    rewritten_query: str | None = Field(
        default=None,
        description="Standalone retrieval query, or null if the latest message is not document-related.",
    )
    confidence: Confidence = Field(description="Confidence in the rewritten query.")
    reasoning: str = Field(description="Short reason for the rewrite.")


class AgentState(TypedDict):
    payload: StreamMessageRequest
    user: User
    model_name: str
    history: list[HumanMessage | AIMessage]
    chat_mode: ChatMode
    route: Route | Literal["needs_router"]
    route_reason: str
    route_confidence: Confidence
    original_query: str
    retrieval_query: str
    rewrite_used: bool
    rewrite_source: RewriteSource
    rewrite_confidence: Confidence
    initial_k: int
    final_k: int
    retrieval_confidence: Confidence
    retrieved_chunk_count: int
    selected_chunk_count: int
    retrieved_document_ids: list[str]
    selected_context: list[dict]
    fallback_reason: str | None
    retrieval_latency_ms: int
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
