import json
import os
import httpx
from langchain.messages import AIMessage, HumanMessage, UsageMetadata
from models.user import User
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from sqlalchemy.orm import Session
from models.message import Message
from respositories.conversation_repository import ConversationRepository
from respositories.message_repository import MessageRepository
from respositories.user_repository import UserRepository
from schemas.models import ModelList, Role, StreamMessageRequest
from langchain_core.documents import Document
from services.rag_service import RagService
from dotenv import load_dotenv
import tiktoken

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
        self.user_repository = UserRepository(self.db)
        self.rag_service = RagService()
        self.models_list = {model.value: get_llm(
            model.value) for model in ModelList}

    @staticmethod
    def _parse_usage(usage: UsageMetadata | None) -> tuple[int, int, int]:
        if not usage:
            return 0, 0, 0
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        return input_tokens, output_tokens, total_tokens

    @staticmethod
    def _estimate_token_count(content: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name="gpt-5")
        output_tokens = encoding.encode(content)
        return len(output_tokens)

    async def stream_messages(self, payload: StreamMessageRequest, user: User):
        system_message = self.rag_service.build_system_message(
            payload.user_content)

        parts: list[str] = []

        model = self.models_list[payload.model_id.value]

        user_input_tokens = 0
        try:
            # Track per-message usage for the user message itself.
            user_input_tokens = int(model.get_num_tokens(payload.user_content))
        except Exception:
            user_input_tokens = 0

        user_message = Message(
            id=payload.message_id,
            role=Role.user,
            content=payload.user_content,
            conversation_id=payload.conversation_id,
            model_id=payload.model_id.value,
            input_tokens=user_input_tokens,
            total_tokens=user_input_tokens,
        )
        self.message_repository.create_message(user_message)

        existing_messages = self.message_repository.get_all_messages(
            payload.conversation_id)
        if (len(existing_messages) == 1):
            self.conversation_repository.update_conversation_title(
                payload.conversation_id,  user_message.content[:10], user.id)

        langchain_messages = [
            HumanMessage(message.content)
            if message.role == Role.user
            else AIMessage(message.content)
            for message in existing_messages
        ]

        langchain_messages = [system_message] + langchain_messages

        langchain_content = "".join(
            [message.content for message in langchain_messages if isinstance(message.content, str)])

        total_assistant_input_tokens = 0
        total_assistant_output_tokens = 0
        total_assistant_total_tokens = 0

        try:
            async for chunk in model.astream(langchain_messages):
                usage = chunk.usage_metadata if hasattr(
                    chunk, "usage_metadata") else None

                if usage:
                    assistant_input_tokens, assistant_output_tokens, assistant_total_tokens = self._parse_usage(
                        usage)
                    total_assistant_input_tokens = assistant_input_tokens
                    total_assistant_output_tokens += assistant_output_tokens
                    total_assistant_total_tokens += assistant_total_tokens
                else:
                    assistant_output_tokens = self._estimate_token_count(getattr(
                        chunk,
                        "text",
                        "",
                    ))
                    total_assistant_input_tokens = self._estimate_token_count(
                        langchain_content)
                    total_assistant_output_tokens += assistant_output_tokens
                    total_assistant_total_tokens = total_assistant_output_tokens + \
                        total_assistant_input_tokens

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
                model_id=payload.model_id.value,
                input_tokens=total_assistant_input_tokens,
                output_tokens=total_assistant_output_tokens,
                total_tokens=total_assistant_total_tokens,
            )

            self.message_repository.create_message(assistant_message)
            self.conversation_repository.increment_usage(
                payload.conversation_id,
                user_input_tokens + total_assistant_input_tokens,
                total_assistant_output_tokens,
                user_input_tokens + total_assistant_total_tokens,
            )
            self.user_repository.increment_usage(
                user.id,
                user_input_tokens + total_assistant_input_tokens,
                total_assistant_output_tokens,
                user_input_tokens + total_assistant_total_tokens,
            )
            self.db.commit()

            yield to_ndjson({"type": "done"})

        except Exception as e:
            self.db.rollback()

            yield to_ndjson(
                {
                    "type": "error",
                    "message": str(e),
                },
            )
