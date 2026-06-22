from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lmnr import Laminar
from app.core.config import settings
from app.core.database import SQL_ENGINE
from app.core.startup import (
    bootstrap_auth_schema,
    bootstrap_document_schema,
    bootstrap_rag_retrieval_event_schema,
)
import app.models.rag_retrieval_event  # noqa: F401
from app.models.base import Base
from app.routers import (
    admin_usage,
    admin_users,
    auth,
    budget_requests,
    chat,
    document_router,
    conversations,
    files,
    messages,
    users,
)


def create_app() -> FastAPI:
    bootstrap_auth_schema(SQL_ENGINE)
    Base.metadata.create_all(bind=SQL_ENGINE)
    bootstrap_document_schema(SQL_ENGINE)
    bootstrap_rag_retrieval_event_schema(SQL_ENGINE)
    Laminar.initialize()

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(budget_requests.router)
    app.include_router(admin_usage.router)
    app.include_router(admin_users.router)
    app.include_router(users.router)
    app.include_router(files.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(messages.router)
    app.include_router(document_router.router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()
