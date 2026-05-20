from sqlalchemy.orm import Session
from models.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def increment_usage(self, user_id: str, input_tokens: int, output_tokens: int, total_tokens: int):
        user = self.db.query(User).where(User.id == user_id).first()
        if not user:
            return None
        user.input_tokens += int(input_tokens)
        user.output_tokens += int(output_tokens)
        user.total_tokens += int(total_tokens)
        self.db.flush()
        return user
