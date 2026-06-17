import asyncio
import json
import unittest
from types import SimpleNamespace

import app.models.budget_request  # noqa: F401
import app.models.conversation  # noqa: F401
import app.models.document_model  # noqa: F401
import app.models.user_permission  # noqa: F401
from app.schemas.chat import ChatMode, ModelList, Role, StreamMessageRequest
from app.services.chat.ndjson import to_ndjson
from app.services.chat.prompt_builder import ChatPromptBuilder
from app.services.chat.route_planner import ChatRoutePlanner
from app.services.chat.tokens import parse_usage
from app.services.chat_stream_service import ChatStreamService


def make_state(
    user_content: str,
    chat_mode: ChatMode = ChatMode.auto,
    available_document_count: int = 1,
    **overrides,
):
    state = {
        "payload": SimpleNamespace(user_content=user_content),
        "user": SimpleNamespace(id="user-1"),
        "history": [],
        "chat_mode": chat_mode,
        "route": "needs_router",
        "route_reason": "",
        "route_confidence": "low",
        "retrieval_query": user_content,
        "rag_docs": [],
        "context": [],
        "answer_kind": "general",
        "citations": [],
        "usage": {
            "router_input_tokens": 0,
            "router_output_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
        },
        "status_events": [],
        "router_used": False,
        "retrieval_used": False,
        "no_document_context": False,
        "available_document_count": available_document_count,
    }
    state.update(overrides)
    return state


class ChatHelperTests(unittest.TestCase):
    def test_to_ndjson_emits_json_line(self):
        result = to_ndjson({"type": "token", "content": "hello"})

        self.assertTrue(result.endswith("\n"))
        self.assertEqual(
            json.loads(result),
            {"type": "token", "content": "hello"},
        )

    def test_parse_usage_handles_missing_and_numeric_values(self):
        self.assertEqual(parse_usage(None), (0, 0, 0))
        self.assertEqual(
            parse_usage(
                {
                    "input_tokens": "3",
                    "output_tokens": 4,
                    "total_tokens": None,
                }
            ),
            (3, 4, 0),
        )


class ChatRoutePlannerTests(unittest.TestCase):
    def test_fast_route_respects_general_only_mode(self):
        planner = ChatRoutePlanner(lambda user: 5)

        result = planner.fast_route(
            make_state("what is in the document?", ChatMode.general_only)
        )

        self.assertEqual(result["route"], "general_chat")
        self.assertEqual(result["route_confidence"], "high")

    def test_fast_route_respects_document_only_mode(self):
        planner = ChatRoutePlanner(lambda user: 5)

        result = planner.fast_route(make_state("hello", ChatMode.document_only))

        self.assertEqual(result["route"], "document_question")
        self.assertEqual(result["status_events"], ["searching_documents"])

    def test_fast_route_uses_document_hint(self):
        planner = ChatRoutePlanner(lambda user: 5)

        result = planner.fast_route(make_state("summarize the pdf"))

        self.assertEqual(result["route"], "document_question")
        self.assertEqual(result["route_reason"], "Message contains document-reference keywords.")

    def test_fast_route_uses_general_path_for_greeting_or_no_documents(self):
        planner = ChatRoutePlanner(lambda user: 0)

        result = planner.fast_route(make_state("hello", available_document_count=0))

        self.assertEqual(result["route"], "general_chat")
        self.assertEqual(result["available_document_count"], 0)

    def test_fast_route_uses_router_for_ambiguous_auto_message(self):
        planner = ChatRoutePlanner(lambda user: 2)

        result = planner.fast_route(make_state("what should I do next?"))

        self.assertEqual(result["route"], "needs_router")
        self.assertEqual(result["route_confidence"], "low")


class ChatPromptBuilderTests(unittest.TestCase):
    def test_prompt_builder_returns_no_document_fallback(self):
        builder = ChatPromptBuilder()

        result = builder.build_answer_prompt(
            make_state(
                "what is in the file?",
                route="document_question",
                no_document_context=True,
                available_document_count=0,
            )
        )

        self.assertEqual(result["answer_kind"], "documents")
        self.assertEqual(result["status_events"], ["no_document_match"])
        self.assertEqual(
            result["response_override"],
            "I do not see any ready uploaded documents to search yet. Please upload a document first, or switch to general mode.",
        )

    def test_prompt_builder_returns_document_prompt_and_citations(self):
        builder = ChatPromptBuilder()

        result = builder.build_answer_prompt(
            make_state(
                "what is in the file?",
                route="document_question",
                context=[
                    {
                        "document_id": "doc-1",
                        "src": "contract.pdf",
                        "page": 2,
                        "text": "Termination requires 30 days notice.",
                    }
                ],
                status_events=["searching_documents"],
            )
        )

        self.assertEqual(result["answer_kind"], "documents")
        self.assertEqual(result["status_events"], ["searching_documents", "answering_from_documents"])
        self.assertEqual(
            result["citations"][0],
            {
                "documentId": "doc-1",
                "fileName": "contract.pdf",
                "page": 2,
                "quote": "Termination requires 30 days notice.",
            },
        )
        self.assertIn("Document context:", result["system_message"].content)


class FakeMessageRepository:
    def __init__(self):
        self.created = []

    def create_message(self, message):
        self.created.append(message)
        return message

    def get_all_messages(self, conversation_id):
        return list(self.created)


class FakeConversationRepository:
    def __init__(self):
        self.titles = []

    def update_conversation_title(self, conversation_id, title, owner_id):
        self.titles.append((conversation_id, title, owner_id))


class FakeGraph:
    def __init__(self, graph_state):
        self.graph_state = graph_state

    def invoke(self, state):
        return dict(self.graph_state)


class FakeDb:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class FakeChunk:
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModel:
    def __init__(self, chunks):
        self.chunks = chunks

    def get_num_tokens(self, content):
        return len(content.split())

    async def astream(self, messages):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class ChatStreamServiceTests(unittest.TestCase):
    def build_service(self, graph_state, model):
        service = ChatStreamService.__new__(ChatStreamService)
        service.db = FakeDb()
        service.message_repository = FakeMessageRepository()
        service.conversation_repository = FakeConversationRepository()
        service.graph = FakeGraph(graph_state)
        service.models_list = {ModelList.gpt_5_nano.value: model}
        return service

    def test_stream_messages_persists_and_streams_expected_events(self):
        graph_state = {
            "status_events": ["answering_generally"],
            "citations": [{"documentId": "doc-1"}],
            "usage": {
                "router_input_tokens": 1,
                "router_output_tokens": 2,
                "answer_input_tokens": 0,
                "answer_output_tokens": 0,
            },
            "system_message": SimpleNamespace(content="You are helpful."),
            "route": "general_chat",
            "route_reason": "test",
            "route_confidence": "high",
            "router_used": False,
            "retrieval_used": False,
            "retrieval_query": "hello",
            "answer_kind": "general",
        }
        service = self.build_service(
            graph_state,
            FakeModel(
                [
                    FakeChunk(
                        "Hello",
                        {
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 8,
                        },
                    ),
                    FakeChunk(" there", {"input_tokens": 5, "output_tokens": 2}),
                ]
            ),
        )
        payload = StreamMessageRequest(
            message_id="msg-1",
            user_content="hello",
            conversation_id="convo-1",
            model_id=ModelList.gpt_5_nano.value,
        )
        user = SimpleNamespace(id="user-1")

        events = asyncio.run(collect_stream(service.stream_messages(payload, user)))
        decoded = [json.loads(event) for event in events]

        self.assertEqual(
            [event["type"] for event in decoded],
            ["status", "citation", "token", "token", "done"],
        )
        self.assertEqual(decoded[2]["content"], "Hello")
        self.assertEqual(decoded[3]["content"], " there")
        self.assertTrue(service.db.committed)
        self.assertEqual(len(service.message_repository.created), 2)
        assistant_message = service.message_repository.created[-1]
        self.assertEqual(assistant_message.content, "Hello there")
        self.assertEqual(assistant_message.input_tokens, 6)
        self.assertEqual(assistant_message.output_tokens, 7)
        self.assertEqual(assistant_message.total_tokens, 13)
        self.assertEqual(assistant_message.payload_json["route"], "general_chat")
        self.assertEqual(assistant_message.payload_json["citation_count"], 1)

    def test_stream_messages_rolls_back_and_emits_error_on_model_failure(self):
        graph_state = {
            "status_events": [],
            "citations": [],
            "usage": {
                "router_input_tokens": 0,
                "router_output_tokens": 0,
                "answer_input_tokens": 0,
                "answer_output_tokens": 0,
            },
            "system_message": SimpleNamespace(content="You are helpful."),
            "route": "general_chat",
            "route_reason": "test",
            "route_confidence": "high",
            "router_used": False,
            "retrieval_used": False,
            "retrieval_query": "hello",
            "answer_kind": "general",
        }
        service = self.build_service(graph_state, FakeModel([RuntimeError("boom")]))
        payload = StreamMessageRequest(
            message_id="msg-1",
            user_content="hello",
            conversation_id="convo-1",
            model_id=ModelList.gpt_5_nano.value,
        )
        user = SimpleNamespace(id="user-1")

        events = asyncio.run(collect_stream(service.stream_messages(payload, user)))
        decoded = [json.loads(event) for event in events]

        self.assertEqual(decoded, [{"type": "error", "message": "boom"}])
        self.assertTrue(service.db.rolled_back)
        self.assertFalse(service.db.committed)


async def collect_stream(stream):
    return [event async for event in stream]


if __name__ == "__main__":
    unittest.main()
