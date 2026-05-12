

from sqlalchemy.orm import Session

from models.conversation import Conversation


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_conversations(self, user_id: str = "1"):
        return self.db.query(Conversation).where(Conversation.owner_id == user_id).order_by(Conversation.updated_at.desc()).all()

    def create_conversation(self, conversation: Conversation):
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_conversation_title(self, conversation_id: str, title: str):
        conversation = self.db.query(Conversation).where(
            Conversation.id == conversation_id).first()
        if not conversation:
            conversation = Conversation(id=conversation_id)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
        conversation.title = title
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
