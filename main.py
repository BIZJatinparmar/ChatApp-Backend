from db import get_db, SQL_ENGINE
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from models.Base import Base
from models.User import User
from schemas.models import Role
from models.Message import Message
from models.Conversation import Conversation
import json
from fastapi.middleware.cors import CORSMiddleware
import dotenv
import os
from schemas.models import MessageCreateRequest,StreamMessageRequest
from langchain.chat_models import init_chat_model
from routers import auth, users
from routers.users import get_current_user
from langchain.messages import HumanMessage, AIMessage, SystemMessage

Base.metadata.create_all(bind=SQL_ENGINE)

dotenv.load_dotenv()
openAiEnvKey =  os.getenv("OPENAI_API_KEY")

if openAiEnvKey is not None:
    os.environ["OPENAI_API_KEY"] =openAiEnvKey


model = init_chat_model(model="gpt-5-nano-2025-08-07")

app = FastAPI()

app.include_router(auth.router)
app.include_router(users.router)

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

def ndjson(event: dict[str,str]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"




@app.post("/chat/stream")
async def chat_stream(
    payload: StreamMessageRequest,
    user:User = Depends(get_current_user),
    db:Session = Depends(get_db)):
        async def event_stream():
            parts: list[str] = []
            message = Message(
                id=payload.message_id,
                role=Role.user,
                content=payload.user_content,
                conversation_id=payload.conversation_id,
                )
            db.add(message)
            db.commit()
            allMessages = db.query(Message).where(Message.conversation_id == payload.conversation_id)
            allMessages = [HumanMessage(message.content) if message.role == Role.user else AIMessage(message.content) for message in allMessages]
            try: 
                async for chunk in model.astream(allMessages):
                    parts.append(chunk.text)
                    yield chunk.text
            except Exception as e:
                yield ndjson({"type": "error", "message": str(e)})
            finally:
                full_response = "".join(parts).strip()
                message = Message(
                    content=full_response,
                    role=Role.assistant,
                    conversation_id=payload.conversation_id
                )
                db.add(message)
                db.commit()
        
        return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-accel-Buffering": "no"
                }
            )


@app.get("/conversations")
def conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Conversation).where(Conversation.owner_id == user.id).all()
    return {
        'conversations': query
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
    return {
        'messages': query
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
    