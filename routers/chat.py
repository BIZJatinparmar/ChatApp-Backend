from fastapi import Depends
from fastapi import APIRouter
from db import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import models
from models.conversation import Conversation
from models.user import User
from schemas.models import ModelList, Role
from models.message import Message
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
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
import httpx

custom_client = httpx.Client(
    verify="C:\\Users\\jatin.parmar\\Documents\\zscaler.crt")

dotenv.load_dotenv()
openAiEnvKey = os.getenv("OPENAI_API_KEY")

if openAiEnvKey is not None:
    os.environ["OPENAI_API_KEY"] = openAiEnvKey


router = APIRouter(prefix="/chat", tags=["chat"])

models_list = {}


def get_llm(model_name: str, temperature: float = 0) -> AzureAIOpenAIApiChatModel:
    return AzureAIOpenAIApiChatModel(
        model=model_name,
        temperature=temperature,
        endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        credential=os.environ["AZURE_API_KEY"],
        http_client=custom_client
    )


for model in ModelList:
    models_list[model.value] = get_llm(model.value)


def ndjson(event: dict[str, str]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def format_context(docs: list[Document]) -> list[dict]:
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
    vector_db = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={
                                       "k": 5, "score_threshold": 0.7})
    docs = retriever.invoke(payload.user_content)
    return format_context(docs)


def to_ndjson(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


@router.post("/stream")
async def chat_stream(
    payload: StreamMessageRequest,
    db: Session = Depends(get_db),
):
    embeddings = OpenAIEmbeddings()
    faiss_path = os.path.join(os.getcwd(), "../faiss_indices")

    if os.path.exists(faiss_path):
        vector_db = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        docs = retriever.invoke(payload.user_content)
        context = format_context(docs)

        system_message_text = """
You are a helpful AI assistant.
Use the following context to answer the user's question.
If the answer is not in context, answer based on your own knowledge.

Context:
"""

        for c in context:
            system_message_text += (
                f"- Source: {c['src']}, Page: {c['page']}\n"
                f"  Text: {c['text']}\n"
            )
    else:
        system_message_text = "You are a helpful AI assistant."

    system_message = SystemMessage(system_message_text)

    async def event_stream():
        parts: list[str] = []

        user_message = Message(
            id=payload.message_id,
            role=Role.user,
            content=payload.user_content,
            conversation_id=payload.conversation_id,
        )

        db.add(user_message)
        db.commit()

        existing_messages = (
            db.query(Message)
            .where(Message.conversation_id == payload.conversation_id)
            .all()
        )
        if (len(existing_messages) == 1):
            db.query(Conversation).where(Conversation.id ==
                                         payload.conversation_id).update({"title": user_message.content[:10]})

        langchain_messages = [
            HumanMessage(message.content)
            if message.role == Role.user
            else AIMessage(message.content)
            for message in existing_messages
        ]

        langchain_messages = [system_message] + langchain_messages

        print(payload.model_id)
        model = models_list[payload.model_id.value]

        try:
            async for chunk in model.astream(langchain_messages):
                content = getattr(
                    chunk,
                    "text",
                    "",
                )

                if not content:
                    continue

                parts.append(content)

                yield to_ndjson(
                    {
                        "type": "token",
                        "content": content,
                    },
                )

            full_response = "".join(parts).strip()

            assistant_message = Message(
                content=full_response,
                role=Role.assistant,
                conversation_id=payload.conversation_id,

            )

            db.add(assistant_message)
            db.commit()

            yield to_ndjson({"type": "done"})

        except Exception as e:
            db.rollback()

            yield to_ndjson(
                {
                    "type": "error",
                    "message": str(e),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
