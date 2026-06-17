from datetime import datetime

from pydantic import BaseModel

from app.schemas.chat import Role


class MessageCreateRequest(BaseModel):
    id: str
    conversationId: str
    role: Role
    content: str


class MessageOut(BaseModel):
    id: str
    content: str
    role: str
    conversationId: str
    createdAt: datetime
    payloadJson: dict
    modelId: str | None
    inputTokens: int
    outputTokens: int
    totalTokens: int


class ConversationMessagesResponse(BaseModel):
    messages: list[MessageOut]
