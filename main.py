from db import get_db, SQL_ENGINE
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, Request
from models.base import Base
from models.user import User
from models.message import Message
from models.conversation import Conversation
from fastapi.middleware.cors import CORSMiddleware
from schemas.models import MessageCreateRequest
from routers import auth, chat_router, conversation_router, message_router, users, file
from routers.users import get_current_user

Base.metadata.create_all(bind=SQL_ENGINE)


app = FastAPI()

app.include_router(auth.router)
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
