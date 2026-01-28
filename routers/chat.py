from fastapi import APIRouter
from db import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from models.User import User
from schemas.models import Role
from models.Message import Message
from schemas.models import StreamMessageRequest
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from .users import get_current_user
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
import os
import json
import dotenv


dotenv.load_dotenv()
openAiEnvKey =  os.getenv("OPENAI_API_KEY")

if openAiEnvKey is not None:
    os.environ["OPENAI_API_KEY"] =openAiEnvKey


router = APIRouter(prefix="/chat", tags=["chat"])


model = init_chat_model(model="gpt-5-nano-2025-08-07", )

def ndjson(event: dict[str,str]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"

def format_context(docs:list[Document])->list[dict]:
    lines = []
    for d in docs:
        data = {
        "src": d.metadata.get("source", "unknown"),
        "page": d.metadata.get("page", "unknown"),
        "text": d.page_content.strip().replace("\n", " ")
        }
        lines.append(data)
    
    return lines
    


@router.post("/docs")
async def get_docs(payload: StreamMessageRequest):
    embeddings = OpenAIEmbeddings()
    faiss_path = os.path.join(os.getcwd(), "../faiss_indices")
    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k":5,"score_threshold":0.7})
    docs = retriever.invoke(payload.user_content)
    return format_context(docs)

@router.post("/stream")
async def chat_stream(
    payload: StreamMessageRequest,
    user:User = Depends(get_current_user),
    db:Session = Depends(get_db)):
        
        embeddings = OpenAIEmbeddings()
        faiss_path = os.path.join(os.getcwd(), "../faiss_indices")
        vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k":4})
        docs = retriever.invoke(payload.user_content)
        context = format_context(docs)
        system_message_text = """You are a helpful AI assistant.
          Use the following context to answer the user's question.
          If The answer is not in context, answer based on your own knowledge.
             \n\n  Context:\n"""
        for c in context:   
            system_message_text += f"- Source: {c['src']}, Page: {c['page']}\n  Text: {c['text']}\n"

        system_message = SystemMessage(system_message_text)


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
            allMessages = db.query(Message).where(Message.conversation_id == payload.conversation_id).all()
            allMessages = [HumanMessage(message.content) if message.role == Role.user else AIMessage(message.content) for message in allMessages]
            allMessages = [system_message] + allMessages
            try: 
                async for chunk in model.astream(allMessages):
                    parts.append(chunk.text)
                    yield ndjson({"type": "token", "content": chunk.text})
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

