from datetime import datetime

from pydantic import BaseModel


class ConversationCreateRequest(BaseModel):
    id: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str | None
    ownerId: str
    createdAt: datetime
    updatedAt: datetime
    inputTokens: int
    outputTokens: int
    totalTokens: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
