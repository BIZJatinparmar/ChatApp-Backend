from typing import Literal, NotRequired, TypedDict

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.models.user import User
from app.schemas.chat import ChatMode, StreamMessageRequest


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
