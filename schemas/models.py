from pydantic import BaseModel
from enum import Enum

class Role(str,Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'

class MessageCreateRequest(BaseModel):
    id: str
    conversationId:str
    role: Role
    content: str


class StreamMessageRequest(BaseModel):
    message_id: str
    user_content: str
    conversation_id:str