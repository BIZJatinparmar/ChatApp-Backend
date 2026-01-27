from sqlalchemy.orm import DeclarativeBase
import uuid

class Base(DeclarativeBase):
    pass


def uuid_str() -> str:
    return str(uuid.uuid4())
