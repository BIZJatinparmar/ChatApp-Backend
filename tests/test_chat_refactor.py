import asyncio
import json
import unittest
from types import SimpleNamespace

from langchain.messages import HumanMessage
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import create_engine, inspect

from scripts.run_rag_evals import check_expectations, evaluate_case

import app.models.budget_request  # noqa: F401
import app.models.conversation  # noqa: F401
import app.models.document_model  # noqa: F401
import app.models.message  # noqa: F401
import app.models.rag_retrieval_event  # noqa: F401
import app.models.user  # noqa: F401
import app.models.user_permission  # noqa: F401
from app.core.startup import bootstrap_rag_retrieval_event_schema
from app.models.base import Base
from app.schemas.chat import ChatMode, ModelList, Role, StreamMessageRequest
from app.services.chat.document_context import ChatDocumentContext
from app.services.chat.ndjson import to_ndjson
from app.services.chat.prompt_builder import ChatPromptBuilder
from app.services.chat.route_planner import ChatRoutePlanner
from app.services.chat.tokens import parse_usage
from app.services.chat.types import QueryRewriteStructure
from app.services.chat_stream_service import (
    ChatStreamService,
    extract_citation_indexes,
    filter_citations_for_answer,
)
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.rag_service import RagService


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
        "original_query": user_content,
        "retrieval_query": user_content,
        "rewrite_used": False,
        "rewrite_source": "none",
        "rewrite_confidence": "low",
        "initial_k": 0,
        "final_k": 0,
        "retrieval_confidence": "low",
        "retrieved_chunk_count": 0,
        "selected_chunk_count": 0,
        "retrieved_document_ids": [],
        "selected_context": [],
        "fallback_reason": None,
        "retrieval_latency_ms": 0,
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

    def test_filter_citations_for_answer_keeps_only_used_indexes(self):
        citations = [
            {"index": 1, "documentId": "doc-1"},
            {"index": 2, "documentId": "doc-2"},
            {"index": 3, "documentId": "doc-3"},
        ]

        self.assertEqual(extract_citation_indexes("Answer [1] and [3](bad-link)"), {1, 3})
        self.assertEqual(
            filter_citations_for_answer("Answer [1] and [3](bad-link)", citations),
            [
                {"index": 1, "documentId": "doc-1"},
                {"index": 3, "documentId": "doc-3"},
            ],
        )


class RagSchemaTests(unittest.TestCase):
    def test_bootstrap_creates_rag_retrieval_events_table(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        bootstrap_rag_retrieval_event_schema(engine)

        inspector = inspect(engine)
        self.assertIn("rag_retrieval_events", inspector.get_table_names())
        columns = {column["name"] for column in inspector.get_columns("rag_retrieval_events")}
        self.assertIn("rewritten_query", columns)
        self.assertIn("selected_context", columns)


class FakeStructuredInvoker:
    def __init__(self, response=None, should_raise=False):
        self.response = response
        self.should_raise = should_raise

    def invoke(self, messages):
        if self.should_raise:
            raise RuntimeError("rewrite failed")
        return {"parsed": self.response, "raw": None, "parsing_error": None}


class FakeRewriteLlm:
    def __init__(self, response=None, should_raise=False):
        self.response = response
        self.should_raise = should_raise

    def with_structured_output(self, *args, **kwargs):
        return FakeStructuredInvoker(self.response, self.should_raise)

    def get_num_tokens(self, text):
        return len(text.split())


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

    def test_fast_route_treats_document_noun_questions_as_document_hints(self):
        planner = ChatRoutePlanner(lambda user: 5)

        examples = [
            "What is in the project proposal?",
            "What does the vendor agreement state?",
            "Summarize the quarterly report.",
            "What does the Nippon SOW state?",
        ]

        for example in examples:
            with self.subTest(example=example):
                result = planner.fast_route(make_state(example))
                self.assertEqual(result["route"], "document_question")
                self.assertEqual(result["route_confidence"], "high")

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

    def test_plan_retrieval_uses_llm_rewrite_for_document_route(self):
        planner = ChatRoutePlanner(
            lambda user: 1,
            llm_factory=lambda model, temperature: FakeRewriteLlm(
                QueryRewriteStructure(
                    rewritten_query="What does the contract say about termination?",
                    confidence="high",
                    reasoning="Resolved follow-up from chat history.",
                )
            ),
        )

        result = planner.plan_retrieval(
            make_state(
                "what about termination?",
                ChatMode.document_only,
                route="document_question",
                route_reason="User selected document-only mode.",
            )
        )

        self.assertEqual(
            result["retrieval_query"],
            "What does the contract say about termination?",
        )
        self.assertTrue(result["rewrite_used"])
        self.assertEqual(result["rewrite_source"], "llm")
        self.assertEqual(result["rewrite_confidence"], "high")
        self.assertEqual(result["initial_k"], 40)

    def test_plan_retrieval_uses_heuristic_when_llm_rewrite_fails(self):
        planner = ChatRoutePlanner(
            lambda user: 1,
            llm_factory=lambda model, temperature: FakeRewriteLlm(should_raise=True),
        )

        result = planner.plan_retrieval(
            make_state(
                "what about termination?",
                ChatMode.auto,
                route="document_question",
                route_reason="Message contains document-reference keywords.",
                history=[HumanMessage(content="We were discussing the vendor agreement.")],
            )
        )

        self.assertIn("what about termination?", result["retrieval_query"])
        self.assertIn("vendor agreement", result["retrieval_query"])
        self.assertTrue(result["rewrite_used"])
        self.assertEqual(result["rewrite_source"], "heuristic")
        self.assertEqual(result["initial_k"], 30)

    def test_heuristic_rewrite_uses_compact_salient_history(self):
        planner = ChatRoutePlanner(
            lambda user: 1,
            llm_factory=lambda model, temperature: FakeRewriteLlm(should_raise=True),
        )

        result = planner.plan_retrieval(
            make_state(
                "What about termination?",
                ChatMode.document_only,
                route="document_question",
                route_reason="User selected document-only mode.",
                history=[
                    HumanMessage(content="We are reviewing the Vendor Agreement."),
                ],
            )
        )

        self.assertEqual(
            result["retrieval_query"],
            "Vendor Agreement What about termination?",
        )
        self.assertNotIn("Context from recent conversation", result["retrieval_query"])

    def test_document_correction_followup_reuses_previous_question(self):
        planner = ChatRoutePlanner(
            lambda user: 1,
            llm_factory=lambda model, temperature: FakeRewriteLlm(
                QueryRewriteStructure(
                    rewritten_query="uploaded document reference",
                    confidence="medium",
                    reasoning="Bad correction rewrite.",
                )
            ),
        )

        result = planner.plan_retrieval(
            make_state(
                "It is an uploaded document you dummy",
                ChatMode.document_only,
                route="document_question",
                route_reason="Message contains document-reference keywords.",
                history=[
                    HumanMessage(content="What does the Nippon SOW state?"),
                    HumanMessage(content="It is an uploaded document you dummy"),
                ],
            )
        )

        self.assertEqual(result["retrieval_query"], "What does the Nippon SOW state?")
        self.assertEqual(result["rewrite_source"], "heuristic")


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
                "index": 1,
                "documentId": "doc-1",
                "fileName": "contract.pdf",
                "page": 2,
                "quote": "Termination requires 30 days notice.",
            },
        )
        self.assertIn("Document context:", result["system_message"].content)
        self.assertIn("Use [1] style markers only", result["system_message"].content)
        self.assertIn("Do not create markdown links", result["system_message"].content)

    def test_prompt_builder_omits_clickable_citations_without_document_id(self):
        builder = ChatPromptBuilder()

        result = builder.build_answer_prompt(
            make_state(
                "what is in the file?",
                route="document_question",
                context=[
                    {
                        "document_id": None,
                        "src": "contract.pdf",
                        "page": 2,
                        "text": "Termination requires 30 days notice.",
                    }
                ],
            )
        )

        self.assertEqual(result["citations"], [])


class DocumentServiceTests(unittest.TestCase):
    def test_pdf_page_metadata_uses_preview_page_numbers(self):
        document = SimpleNamespace(
            filename="contract.pdf",
            owner_id="user-1",
            id="doc-1",
            filetype="pdf",
        )

        metadata = DocumentService._metadata(document, page=1)

        self.assertEqual(metadata["page"], "1")

    def test_chunk_metadata_adds_stable_chunk_id_and_start_index(self):
        document = SimpleNamespace(id="doc-1")
        chunks = [
            LangchainDocument(
                page_content="First chunk",
                metadata={"document_id": "doc-1", "start_index": 12},
            )
        ]

        result = DocumentService._with_chunk_metadata(document, chunks)

        self.assertEqual(result[0].metadata["chunk_id"], "doc-1:0")
        self.assertEqual(result[0].metadata["start_index"], "12")


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, query, k, filter):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.results[:k]


class RagServiceTests(unittest.TestCase):
    def test_ranked_context_uses_owner_filter_and_dedupes_chunks(self):
        docs = [
            (
                LangchainDocument(
                    page_content="Termination requires 30 days notice.",
                    metadata={
                        "document_id": "doc-1",
                        "filename": "contract.pdf",
                        "page": "2",
                        "chunk_id": "doc-1:1",
                    },
                ),
                0.2,
            ),
            (
                LangchainDocument(
                    page_content="Termination requires 30 days notice.",
                    metadata={
                        "document_id": "doc-1",
                        "filename": "contract.pdf",
                        "page": "2",
                        "chunk_id": "doc-1:1",
                    },
                ),
                0.21,
            ),
            (
                LangchainDocument(
                    page_content="Payment is due in 15 days.",
                    metadata={
                        "document_id": "doc-1",
                        "filename": "contract.pdf",
                        "page": "3",
                        "chunk_id": "doc-1:2",
                    },
                ),
                0.3,
            ),
        ]
        vector_store = FakeVectorStore(docs)
        service = RagService(vector_store)

        result = service.get_ranked_context_result(
            "What does the contract say about termination?",
            SimpleNamespace(id="user-1"),
            initial_k=10,
            final_k=8,
        )

        self.assertEqual(vector_store.calls[0]["filter"], {"owner_id": "user-1"})
        self.assertEqual(len(result.retrieved_docs), 3)
        self.assertEqual(len(result.selected_docs), 2)
        self.assertEqual(result.retrieval_confidence, "high")
        self.assertEqual(result.retrieved_document_ids, ["doc-1"])
        self.assertEqual(result.selected_context[0]["chunk_id"], "doc-1:1")

    def test_document_context_low_confidence_triggers_no_context(self):
        vector_store = FakeVectorStore([])
        context = ChatDocumentContext.__new__(ChatDocumentContext)
        context.db = None
        context.rag_service = RagService(vector_store)

        result = context.retrieve_documents(
            make_state(
                "missing answer",
                route="document_question",
                available_document_count=1,
                initial_k=20,
                final_k=8,
            )
        )

        self.assertTrue(result["retrieval_used"])
        self.assertTrue(result["no_document_context"])
        self.assertEqual(result["retrieval_confidence"], "low")
        self.assertEqual(result["fallback_reason"], "no_retrieval_results")

    def test_reranker_prefers_source_and_topic_match(self):
        docs = [
            (
                LangchainDocument(
                    page_content="Statement of Work abstract for a generic AI platform.",
                    metadata={
                        "document_id": "abg",
                        "filename": "ABG_platform.pdf",
                        "page": "1",
                        "chunk_id": "abg:1",
                    },
                ),
                0.1,
            ),
            (
                LangchainDocument(
                    page_content="Resource Allocation Role FTE Lead Gen AI Engineer PMO.",
                    metadata={
                        "document_id": "wipro",
                        "filename": "Wipro_invoice_processing.pdf",
                        "page": "13",
                        "chunk_id": "wipro:13",
                    },
                ),
                0.35,
            ),
        ]
        service = RagService(FakeVectorStore(docs))

        result = service.get_ranked_context_result(
            "Wipro invoice processing resource allocation roles FTE",
            SimpleNamespace(id="user-1"),
            initial_k=10,
            final_k=2,
        )

        self.assertEqual(result.selected_docs[0].metadata["document_id"], "wipro")
        self.assertGreater(result.selected_docs[0].metadata["rerank_score"], 0)

    def test_reranker_limits_foreign_source_contamination(self):
        docs = [
            (
                LangchainDocument(
                    page_content=f"Wipro invoice validation reconciliation control {index}",
                    metadata={
                        "document_id": "wipro",
                        "filename": "Wipro_invoice_processing.pdf",
                        "page": str(index),
                        "chunk_id": f"wipro:{index}",
                    },
                ),
                0.3 + (index * 0.01),
            )
            for index in range(1, 5)
        ] + [
            (
                LangchainDocument(
                    page_content="Tata innovation platform validation overview.",
                    metadata={
                        "document_id": "tata",
                        "filename": "Tata_innovation.pdf",
                        "page": "3",
                        "chunk_id": "tata:3",
                    },
                ),
                0.2,
            )
        ]
        service = RagService(FakeVectorStore(docs))

        result = service.get_ranked_context_result(
            "Wipro invoice validation reconciliation",
            SimpleNamespace(id="user-1"),
            initial_k=10,
            final_k=4,
        )

        self.assertEqual(
            [doc.metadata["document_id"] for doc in result.selected_docs],
            ["wipro", "wipro", "wipro", "wipro"],
        )


class FakeEvalDocumentContext:
    def __init__(self):
        self.queries = []

    def available_document_count(self, user):
        return 1

    def retrieve_documents(self, state):
        self.queries.append(state["retrieval_query"])
        return {
            "rag_docs": [],
            "context": [
                {
                    "document_id": "doc-1",
                    "src": "contract.pdf",
                    "page": "2",
                    "text": "Vendor agreement termination requires 30 days notice.",
                }
            ],
            "retrieval_used": True,
            "no_document_context": False,
            "retrieval_confidence": "high",
            "retrieved_chunk_count": 1,
            "selected_chunk_count": 1,
            "retrieved_document_ids": ["doc-1"],
            "selected_context": [
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1:1",
                    "filename": "contract.pdf",
                    "page": "2",
                    "quote": "Vendor agreement termination requires 30 days notice.",
                    "rank": 1,
                    "score": 0.2,
                }
            ],
            "fallback_reason": None,
            "retrieval_latency_ms": 1,
        }


class RagEvalRunnerTests(unittest.TestCase):
    def test_eval_case_requires_user_content(self):
        result = evaluate_case(
            case={"name": "legacy_case", "query": "direct vector query", "expected": {}},
            defaults={},
            route_planner=ChatRoutePlanner(lambda user: 1),
            document_context=FakeEvalDocumentContext(),
            default_user_id="user-1",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].check, "user_content")

    def test_query_expectation_failures_are_reported(self):
        state = make_state(
            "What about resources?",
            route="general_chat",
            route_confidence="low",
            retrieval_query="Tata resource allocation",
            rewrite_source="none",
            rewrite_confidence="low",
            retrieval_confidence="medium",
            context=[{"text": "Lead Gen AI Engineer"}],
        )

        failures = check_expectations(
            expected={
                "route": "document_question",
                "rewrite_source": "llm",
                "min_rewrite_confidence": "medium",
                "generated_query_terms_all": ["Tata", "timeline"],
                "generated_query_forbidden_terms_any": ["allocation"],
            },
            state=state,
            selected_doc_ids=[],
            retrieved_doc_ids=[],
            selected_pages_=[],
            selected_sources=[],
            selected_source_sequence_=[],
            selected_text="Lead Gen AI Engineer",
        )

        self.assertEqual(
            {failure.check for failure in failures},
            {
                "route",
                "rewrite_source",
                "min_rewrite_confidence",
                "generated_query_terms_all",
                "generated_query_forbidden_terms_any",
            },
        )

    def test_eval_runs_rewrite_before_retrieval(self):
        document_context = FakeEvalDocumentContext()
        planner = ChatRoutePlanner(
            document_context.available_document_count,
            llm_factory=lambda model, temperature: FakeRewriteLlm(
                QueryRewriteStructure(
                    rewritten_query="Vendor agreement termination notice",
                    confidence="high",
                    reasoning="Resolved follow-up from chat history.",
                )
            ),
        )

        result = evaluate_case(
            case={
                "name": "termination_followup",
                "chat_mode": "document_only",
                "history": [
                    {
                        "role": "user",
                        "content": "Summarize the vendor agreement.",
                    },
                    {
                        "role": "assistant",
                        "content": "It covers payment and termination clauses.",
                    },
                ],
                "user_content": "What about termination?",
                "expected": {
                    "route": "document_question",
                    "rewrite_source": "llm",
                    "min_rewrite_confidence": "high",
                    "generated_query_terms_all": ["Vendor agreement", "termination"],
                    "selected_text_term_groups_all": [
                        {
                            "name": "termination_notice",
                            "terms": ["30 days notice", "termination requires"],
                        }
                    ],
                    "selected_source_names_forbidden_any": ["policy"],
                    "max_foreign_source_count": 0,
                    "min_confidence": "high",
                    "fallback_reason": None,
                },
            },
            defaults={},
            route_planner=planner,
            document_context=document_context,
            default_user_id="user-1",
        )

        self.assertTrue(result.passed, result.failures)
        self.assertEqual(document_context.queries, ["Vendor agreement termination notice"])
        self.assertEqual(result.summary["user_content"], "What about termination?")
        self.assertEqual(result.summary["retrieval_query"], "Vendor agreement termination notice")
        self.assertEqual(result.summary["selected_document_ids"], ["doc-1"])

    def test_generalized_text_expectations_report_failures(self):
        state = make_state(
            "What about support?",
            route="document_question",
            route_confidence="high",
            retrieval_query="Product policy support tiers",
            rewrite_source="llm",
            rewrite_confidence="high",
            retrieval_confidence="high",
        )

        failures = check_expectations(
            expected={
                "selected_text_terms_all": ["support tiers"],
                "selected_text_terms_any": ["gold", "platinum"],
                "selected_text_term_groups_all": [
                    {"name": "sla", "terms": ["response time", "uptime"]}
                ],
                "selected_source_names_forbidden_any": ["invoice"],
                "max_foreign_source_count": 0,
            },
            state=state,
            selected_doc_ids=[],
            retrieved_doc_ids=[],
            selected_pages_=[],
            selected_sources=["product_policy.txt", "invoice_process.txt"],
            selected_source_sequence_=["product_policy.txt", "invoice_process.txt"],
            selected_text="Product policy support tiers includes silver tier.",
        )

        self.assertEqual(
            {failure.check for failure in failures},
            {
                "selected_text_terms_any",
                "selected_text_term_groups_all",
                "selected_source_names_forbidden_any",
                "max_foreign_source_count",
            },
        )


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


class FakeConversationReadRepository:
    def __init__(self, conversation, messages):
        self.conversation = conversation
        self.messages = messages

    def get_by_id(self, conversation_id):
        return self.conversation

    def get_conversation_messages(self, conversation_id):
        return self.messages


class FakeGraph:
    def __init__(self, graph_state):
        self.graph_state = graph_state

    def invoke(self, state):
        return dict(self.graph_state)


class FakeDb:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.added = []

    def add(self, item):
        self.added.append(item)

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
            "citations": [
                {"index": 1, "documentId": "doc-1"},
                {"index": 2, "documentId": "doc-2"},
                {"index": 3, "documentId": "doc-3"},
            ],
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
            "original_query": "hello",
            "retrieval_query": "hello",
            "rewrite_used": False,
            "rewrite_source": "none",
            "rewrite_confidence": "low",
            "retrieval_confidence": "low",
            "fallback_reason": None,
            "answer_kind": "general",
        }
        service = self.build_service(
            graph_state,
            FakeModel(
                [
                    FakeChunk(
                        "Hello [1]",
                        {
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 8,
                        },
                    ),
                    FakeChunk(" there [3]", {"input_tokens": 5, "output_tokens": 2}),
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
            ["status", "token", "token", "citation", "citation", "done"],
        )
        self.assertEqual(decoded[1]["content"], "Hello [1]")
        self.assertEqual(decoded[2]["content"], " there [3]")
        self.assertEqual(decoded[3]["citation"], {"index": 1, "documentId": "doc-1"})
        self.assertEqual(decoded[4]["citation"], {"index": 3, "documentId": "doc-3"})
        self.assertTrue(service.db.committed)
        self.assertEqual(len(service.message_repository.created), 2)
        assistant_message = service.message_repository.created[-1]
        self.assertEqual(assistant_message.content, "Hello [1] there [3]")
        self.assertEqual(assistant_message.input_tokens, 6)
        self.assertEqual(assistant_message.output_tokens, 7)
        self.assertEqual(assistant_message.total_tokens, 13)
        self.assertEqual(assistant_message.payload_json["route"], "general_chat")
        self.assertEqual(
            assistant_message.payload_json["citations"],
            [
                {"index": 1, "documentId": "doc-1"},
                {"index": 3, "documentId": "doc-3"},
            ],
        )
        self.assertEqual(assistant_message.payload_json["citation_count"], 2)

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
            "original_query": "hello",
            "retrieval_query": "hello",
            "rewrite_used": False,
            "rewrite_source": "none",
            "rewrite_confidence": "low",
            "retrieval_confidence": "low",
            "fallback_reason": None,
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


class ConversationServiceTests(unittest.TestCase):
    def test_list_messages_serializes_payload_json(self):
        service = ConversationService.__new__(ConversationService)
        service.conversations = FakeConversationReadRepository(
            SimpleNamespace(id="convo-1", owner_id="user-1"),
            [
                SimpleNamespace(
                    id="msg-1",
                    content="Answer [1]",
                    role="assistant",
                    conversation_id="convo-1",
                    created_at="2026-06-17T00:00:00",
                    payload_json={"citations": [{"index": 1, "documentId": "doc-1"}]},
                    model_id="gpt-5-nano",
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                )
            ],
        )

        result = service.list_messages("convo-1", SimpleNamespace(id="user-1"))

        self.assertEqual(
            result.messages[0].payloadJson,
            {"citations": [{"index": 1, "documentId": "doc-1"}]},
        )


async def collect_stream(stream):
    return [event async for event in stream]


if __name__ == "__main__":
    unittest.main()
