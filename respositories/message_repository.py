from sqlalchemy.orm import Session
from models.message import Message


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_messages(self, conversation_id):
        return self.db.query(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.desc()).all()

    def create_message(self, message: Message):
        self.db.add(message)
        self.db.flush()
        self.db.refresh(message)
        return message
