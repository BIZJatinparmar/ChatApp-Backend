from db import get_db, SQL_ENGINE
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, Request
from models.Base import Base
from models.User import User
from models.Message import Message
from models.Conversation import Conversation
from fastapi.middleware.cors import CORSMiddleware
from schemas.models import MessageCreateRequest
from routers import auth, users, file, chat
from routers.users import get_current_user

Base.metadata.create_all(bind=SQL_ENGINE)


app = FastAPI()

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(file.router)
app.include_router(chat.router)
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


@app.get("/conversations")
def conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.updated_at.desc()).all()
    transformed = [{"id": data.id, "title": data.title, "ownerId": data.owner_id, "createdAt": data.created_at, "updatedAt": data.updated_at} for data in query]

    return {
        'conversations': transformed
    }

@app.post("/messages")
async def insert_message(message:MessageCreateRequest, db:Session=Depends(get_db)):
    message_db = Message(
        id=message.id,
        content=message.content, 
        conversation_id=message.conversationId,
        role=message.role
        )
    db.add(message_db)
    db.commit()
    db.refresh(message_db)
    return {
        'message': message_db
    }

@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id:str, db: Session = Depends(get_db)):
    query =  db.query(Message).where(Message.conversation_id ==conversation_id).all()
    transformed = [{"id": data.id, "content": data.content, "role": data.role, "conversationId": data.conversation_id, "createdAt": data.created_at} for data in query]
    return {
        'messages': transformed
    }

@app.post("/conversations")
async def create_conversation(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    body = await request.json()
    convo = Conversation(id=body['id'],owner_id=user.id,title="NewChat")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo
    