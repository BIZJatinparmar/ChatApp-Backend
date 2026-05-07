from pydantic import BaseModel
from enum import Enum


class ModelList(str, Enum):
    gpt_5_nano = 'gpt-5-nano'
    gpt_5_4_nano_1 = 'gpt-5.4-nano-1'


class Role(str, Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'


class MessageCreateRequest(BaseModel):
    id: str
    conversationId: str
    role: Role
    content: str


class StreamMessageRequest(BaseModel):
    message_id: str
    user_content: str
    conversation_id: str
    model_id: ModelList
