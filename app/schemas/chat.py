from enum import Enum

from pydantic import BaseModel


class ModelList(str, Enum):
    gpt_5_nano = "gpt-5-nano"
    gpt_5_4_nano_1 = "gpt-5.4-nano-1"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMode(str, Enum):
    auto = "auto"
    document_only = "document_only"
    general_only = "general_only"


class StreamMessageRequest(BaseModel):
    message_id: str
    user_content: str
    conversation_id: str
    model_id: str
    chat_mode: ChatMode = ChatMode.auto
