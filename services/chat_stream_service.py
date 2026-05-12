import json
import os
import httpx
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session
from models.message import Message
from respositories.conversation_repository import ConversationRepository
from respositories.message_repository import MessageRepository
from schemas.models import ModelList, Role, StreamMessageRequest
from langchain_core.documents import Document
from services.rag_service import RagService
from dotenv import load_dotenv

load_dotenv()

custom_client = httpx.Client(
    verify="C:\\Users\\jatin.parmar\\Documents\\zscaler.crt")


def to_ndjson(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


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


def get_llm(model_name: str, temperature: float = 0) -> AzureAIOpenAIApiChatModel:
    return AzureAIOpenAIApiChatModel(
        model=model_name,
        temperature=temperature,
        endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        credential=os.environ["AZURE_API_KEY"],
        http_client=custom_client
    )


class ChatStreamService:

    models_list = {}

    def __init__(self, db: Session):
        self.db = db
        self.message_repository = MessageRepository(self.db)
        self.conversation_repository = ConversationRepository(
            self.db)
        self.rag_service = RagService()
        self.models_list = {model.value: get_llm(
            model.value) for model in ModelList}

    async def stream_messages(self, payload: StreamMessageRequest):
        system_message = self.rag_service.build_system_message(
            payload.user_content)

        parts: list[str] = []

        user_message = Message(
            id=payload.message_id,
            role=Role.user,
            content=payload.user_content,
            conversation_id=payload.conversation_id,
        )
        self.message_repository.create_message(user_message)

        existing_messages = self.message_repository.get_all_messages(
            payload.conversation_id)
        if (len(existing_messages) == 1):
            self.conversation_repository.update_conversation_title(
                payload.conversation_id,  user_message.content[:10])

        langchain_messages = [
            HumanMessage(message.content)
            if message.role == Role.user
            else AIMessage(message.content)
            for message in existing_messages
        ]

        langchain_messages = [system_message] + langchain_messages

        model = self.models_list[payload.model_id.value]

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

            self.message_repository.create_message(assistant_message)

            yield to_ndjson({"type": "done"})

        except Exception as e:
            self.db.rollback()

            yield to_ndjson(
                {
                    "type": "error",
                    "message": str(e),
                },
            )
