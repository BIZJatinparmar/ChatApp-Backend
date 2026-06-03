from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import SQL_ENGINE
from db_bootstrap import bootstrap_auth_schema
from models.base import Base
from routers import auth, budget_requests, chat_router, conversation_router, message_router, users, file
from routers.admin_usage import router as admin_usage_router
from routers.admin_users import router as admin_users_router

bootstrap_auth_schema(SQL_ENGINE)
Base.metadata.create_all(bind=SQL_ENGINE)


app = FastAPI()

app.include_router(auth.router)
app.include_router(budget_requests.router)
app.include_router(admin_usage_router)
app.include_router(admin_users_router)
app.include_router(users.router)
app.include_router(file.router)
app.include_router(chat_router.router)
app.include_router(conversation_router.router)
app.include_router(message_router.router)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
