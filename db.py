from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session,sessionmaker


SQLite_PATH = "sqlite:///./app.db"
SQL_ENGINE = create_engine(SQLite_PATH,echo=True,connect_args={"check_same_thread": False},)
SessionLocal = sessionmaker(bind=SQL_ENGINE, autoflush=False, autocommit=False)

def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
    