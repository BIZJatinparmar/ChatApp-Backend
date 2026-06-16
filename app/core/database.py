from __future__ import annotations

from collections.abc import Generator

from pydantic import SecretStr

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from langchain_postgres import PGEngine, PGVectorStore
from langchain_openai import AzureOpenAIEmbeddings
import os
vector_store = None

SQLite_PATH = "sqlite:///./app.db"
SQL_ENGINE = create_engine(SQLite_PATH, echo=False, connect_args={
                           "check_same_thread": False},)
SessionLocal = sessionmaker(bind=SQL_ENGINE, autoflush=False, autocommit=False)
VECTORSTORE_TABLE_NAME = "user_documents"


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def _vectorstore_table_exists(pg_engine: PGEngine) -> bool:
    async with pg_engine._pool.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                )
                """
            ),
            {"table_name": VECTORSTORE_TABLE_NAME},
        )
        return bool(result.scalar())


def get_vector_db() -> PGVectorStore:

    global vector_store
    if vector_store is not None:
        return vector_store

    pg_engine = PGEngine.from_connection_string(
        f"postgresql+asyncpg://{os.environ['VECTOR_DB_USERNAME']}:{os.environ['VECTOR_DB_PASSWORD']}@{os.environ['VECTOR_DB_HOST']}:{os.environ['VECTOR_DB_PORT']}/{os.environ['VECTOR_DB_NAME']}")

    embeddings = AzureOpenAIEmbeddings(
        api_key=SecretStr(os.environ["AZURE_EMBEDDING_OPENAI_KEY"]),
        azure_endpoint=os.environ["AZURE_EMBEDDING_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_EMBEDDING_OPENAI_DEPLOYMENT_NAME"],
    )
    if not pg_engine._run_as_sync(_vectorstore_table_exists(pg_engine)):
        pg_engine.init_vectorstore_table(
            VECTORSTORE_TABLE_NAME,
            vector_size=1536,
        )

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=VECTORSTORE_TABLE_NAME,
        embedding_service=embeddings
    )
    return vector_store
