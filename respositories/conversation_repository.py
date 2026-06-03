

from sqlalchemy.orm import Session

from models.conversation import Conversation
from models.message import Message


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_conversations(self, user_id: str):
        return self.db.query(Conversation).where(Conversation.owner_id == user_id).order_by(Conversation.updated_at.desc()).all()

    def create_conversation(self, conversation: Conversation):
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_by_id(self, conversation_id: str):
        return self.db.query(Conversation).where(Conversation.id == conversation_id).first()

    def get_conversation_messages(self, conversation_id: str):
        return self.db.query(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).all()

    def update_conversation_title(self, conversation_id: str, title: str, owner_id: str):
        conversation = self.db.query(Conversation).where(
            Conversation.id == conversation_id).first()
        if not conversation:
            conversation = Conversation(id=conversation_id, owner_id=owner_id)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
        conversation.title = title
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def increment_usage(self, conversation_id: str, input_tokens: int, output_tokens: int, total_tokens: int):
        conversation = self.db.query(Conversation).where(
            Conversation.id == conversation_id).first()
        if not conversation:
            return None
        conversation.input_tokens += int(input_tokens)
        conversation.output_tokens += int(output_tokens)
        conversation.total_tokens += int(total_tokens)
        self.db.flush()
        return conversation
