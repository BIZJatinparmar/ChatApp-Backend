from __future__ import annotations

from dotenv import load_dotenv

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document as LangchainDocument
from langchain.messages import AIMessage, HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.database import SessionLocal, get_vector_db  # noqa
import app.models  # noqa: F401
from app.schemas.chat import ChatMode, ModelList, StreamMessageRequest  # noqa
from app.services.chat.document_context import ChatDocumentContext  # noqa
from app.services.chat.route_planner import ChatRoutePlanner  # noqa
from app.services.chat.tokens import empty_usage  # noqa
from app.services.chat.types import AgentState  # noqa
from app.services.rag_service import RagService  # noqa


load_dotenv(PROJECT_ROOT / ".env")


CONFIDENCE_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass(frozen=True)
class EvalFailure:
    check: str
    expected: Any
    actual: Any


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    failures: list[EvalFailure]
    summary: dict[str, Any]


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise ValueError(
            "Eval file must contain either a list of cases or an object with a cases array."
        )
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Eval file object must include a cases array.")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object when provided.")
    fixtures = data.get("fixtures") or {}
    if isinstance(fixtures, dict) and fixtures.get("documents"):
        defaults = dict(defaults)
        defaults["_fixture_documents"] = fixtures["documents"]
    return defaults, cases


class FixtureVectorStore:
    def __init__(self, documents: list[LangchainDocument]):
        self.documents = documents

    def similarity_search_with_score(self, query: str, k: int, filter: dict) -> list[tuple[LangchainDocument, float]]:
        owner_id = filter.get("owner_id")
        query_terms = {
            term
            for term in normalize_text(query).replace("_", " ").split()
            if len(term) >= 3
        }
        scored: list[tuple[LangchainDocument, float]] = []
        for doc in self.documents:
            if owner_id and doc.metadata.get("owner_id") != owner_id:
                continue
            text = normalize_text(
                " ".join(
                    [
                        doc.page_content,
                        str(doc.metadata.get("filename", "")),
                        str(doc.metadata.get("source", "")),
                    ]
                )
            )
            overlap = sum(1 for term in query_terms if term in text)
            score = 1.0 / (1.0 + overlap)
            scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1])[:k]


class FixtureDocumentContext(ChatDocumentContext):
    def __init__(self, fixture_documents: list[dict[str, Any]], default_user_id: str):
        self.db = None
        self.documents = [
            LangchainDocument(
                page_content=str(item.get("text") or ""),
                metadata={
                    "owner_id": str(item.get("owner_id") or default_user_id),
                    "document_id": str(item.get("document_id") or f"fixture-{index}"),
                    "filename": str(item.get("filename") or f"fixture-{index}.txt"),
                    "source": str(item.get("filename") or f"fixture-{index}.txt"),
                    "page": str(item.get("page") or "1"),
                    "chunk_id": str(item.get("chunk_id") or f"fixture-{index}:1"),
                },
            )
            for index, item in enumerate(fixture_documents, start=1)
            if str(item.get("text") or "").strip()
        ]
        self.rag_service = RagService(FixtureVectorStore(self.documents))

    def available_document_count(self, user: Any) -> int:
        return sum(1 for doc in self.documents if doc.metadata.get("owner_id") == user.id)


class FailingStructuredInvoker:
    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        raise RuntimeError("LLM rewrite disabled for fixture evals.")


class FailingLlm:
    def with_structured_output(self, *args: Any, **kwargs: Any) -> FailingStructuredInvoker:
        return FailingStructuredInvoker()

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())


def failing_llm_factory(model: str, temperature: float) -> FailingLlm:
    return FailingLlm()


def normalize_text(value: Any) -> str:
    return str(value or "").casefold()


def selected_document_ids(selected_context: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in selected_context:
        document_id = item.get("document_id")
        if document_id and document_id not in ids:
            ids.append(document_id)
    return ids


def selected_pages(selected_context: list[dict[str, Any]]) -> list[str]:
    pages: list[str] = []
    for item in selected_context:
        page = item.get("page")
        if page is not None and str(page) not in pages:
            pages.append(str(page))
    return pages


def selected_source_names(selected_context: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for item in selected_context:
        source = item.get("filename") or item.get("src")
        if source and str(source) not in sources:
            sources.append(str(source))
    return sources


def selected_source_sequence(selected_context: list[dict[str, Any]]) -> list[str]:
    return [
        str(source)
        for item in selected_context
        if (source := item.get("filename") or item.get("src"))
    ]


def messages_from_case(history: Any) -> list[HumanMessage | AIMessage]:
    if history is None:
        return []
    if not isinstance(history, list):
        raise ValueError("history must be an array of message objects.")

    messages: list[HumanMessage | AIMessage] = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            raise ValueError(f"history[{index}] must be an object.")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"history[{index}].content must be a non-empty string.")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            raise ValueError(
                f"history[{index}].role must be either 'user' or 'assistant'."
            )
    return messages


def initial_state(
    *,
    payload: StreamMessageRequest,
    user: Any,
    history: list[HumanMessage | AIMessage],
    model_name: str,
) -> AgentState:
    return {
        "payload": payload,
        "user": user,
        "model_name": model_name,
        "history": history,
        "chat_mode": payload.chat_mode,
        "route": "needs_router",
        "route_reason": "",
        "route_confidence": "low",
        "original_query": payload.user_content,
        "retrieval_query": payload.user_content,
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
        "system_message": SystemMessage("You are a helpful AI assistant."),
        "answer_kind": "general",
        "citations": [],
        "usage": empty_usage(),
        "status_events": [],
        "router_used": False,
        "retrieval_used": False,
        "no_document_context": False,
        "available_document_count": 0,
    }


def run_chat_retrieval_flow(
    *,
    payload: StreamMessageRequest,
    user: Any,
    history: list[HumanMessage | AIMessage],
    route_planner: ChatRoutePlanner,
    document_context: ChatDocumentContext,
    model_name: str,
) -> AgentState:
    state = initial_state(
        payload=payload,
        user=user,
        history=history,
        model_name=model_name,
    )

    state.update(route_planner.fast_route(state))
    if route_planner.after_route(state) == "router":
        state.update(route_planner.router_and_rewrite(state))
    if route_planner.after_route(state) == "plan":
        state.update(route_planner.plan_retrieval(state))
    if route_planner.after_plan(state) == "retrieve":
        state.update(document_context.retrieve_documents(state))

    return state


def evaluate_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    route_planner: ChatRoutePlanner,
    document_context: ChatDocumentContext,
    default_user_id: str | None,
) -> EvalResult:
    name = str(case.get("name") or case.get("id") or case.get("user_content") or "unnamed")
    user_content = case.get("user_content")
    if not isinstance(user_content, str) or not user_content.strip():
        return EvalResult(
            name=name,
            passed=False,
            failures=[EvalFailure("user_content", "non-empty string", user_content)],
            summary={},
        )

    user_id = case.get("user_id") or defaults.get("user_id") or default_user_id
    if not user_id:
        return EvalResult(
            name=name,
            passed=False,
            failures=[EvalFailure("user_id", "case/defaults/--user-id", None)],
            summary={},
        )

    expected = case.get("expected") or {}
    if not isinstance(expected, dict):
        return EvalResult(
            name=name,
            passed=False,
            failures=[EvalFailure("expected", "object", expected)],
            summary={},
        )

    try:
        chat_mode = ChatMode(case.get("chat_mode") or defaults.get("chat_mode") or ChatMode.auto.value)
    except ValueError:
        return EvalResult(
            name=name,
            passed=False,
            failures=[
                EvalFailure(
                    "chat_mode",
                    [mode.value for mode in ChatMode],
                    case.get("chat_mode"),
                )
            ],
            summary={},
        )

    model_name = str(case.get("model_id") or defaults.get("model_id") or ModelList.gpt_5_nano.value)
    try:
        history = messages_from_case(case.get("history", defaults.get("history", [])))
    except ValueError as exc:
        return EvalResult(
            name=name,
            passed=False,
            failures=[EvalFailure("history", "valid message array", str(exc))],
            summary={},
        )

    payload = StreamMessageRequest(
        message_id=str(case.get("message_id") or f"eval-{name}"),
        user_content=user_content,
        conversation_id=str(case.get("conversation_id") or defaults.get("conversation_id") or "eval-conversation"),
        model_id=model_name,
        chat_mode=chat_mode,
    )
    user = SimpleNamespace(id=str(user_id))

    state = run_chat_retrieval_flow(
        payload=payload,
        user=user,
        history=history,
        route_planner=route_planner,
        document_context=document_context,
        model_name=model_name,
    )

    selected_context = state["selected_context"]
    actual_selected_doc_ids = selected_document_ids(selected_context)
    actual_pages = selected_pages(selected_context)
    actual_sources = selected_source_names(selected_context)
    actual_source_sequence = selected_source_sequence(selected_context)
    selected_text = " ".join(
        normalize_text(item.get("text"))
        for item in state["context"]
    )
    failures = check_expectations(
        expected=expected,
        state=state,
        selected_doc_ids=actual_selected_doc_ids,
        retrieved_doc_ids=state["retrieved_document_ids"],
        selected_pages_=actual_pages,
        selected_sources=actual_sources,
        selected_source_sequence_=actual_source_sequence,
        selected_text=selected_text,
    )

    summary = {
        "user_content": user_content,
        "user_id": str(user_id),
        "chat_mode": chat_mode.value,
        "history_count": len(history),
        "route": state["route"],
        "route_reason": state["route_reason"],
        "route_confidence": state["route_confidence"],
        "router_used": state["router_used"],
        "original_query": state["original_query"],
        "retrieval_query": state["retrieval_query"],
        "rewrite_used": state["rewrite_used"],
        "rewrite_source": state["rewrite_source"],
        "rewrite_confidence": state["rewrite_confidence"],
        "initial_k": state["initial_k"],
        "final_k": state["final_k"],
        "retrieval_used": state["retrieval_used"],
        "retrieval_confidence": state["retrieval_confidence"],
        "fallback_reason": state["fallback_reason"],
        "retrieved_chunk_count": state["retrieved_chunk_count"],
        "selected_chunk_count": state["selected_chunk_count"],
        "retrieved_document_ids": state["retrieved_document_ids"],
        "selected_document_ids": actual_selected_doc_ids,
        "selected_pages": actual_pages,
        "selected_sources": actual_sources,
        "selected_context": selected_context,
    }
    return EvalResult(
        name=name,
        passed=not failures,
        failures=failures,
        summary=summary,
    )


def check_min_confidence(
    *,
    failures: list[EvalFailure],
    check: str,
    expected: Any,
    actual: str,
) -> None:
    if not expected:
        return
    actual_rank = CONFIDENCE_ORDER.get(actual, -1)
    expected_rank = CONFIDENCE_ORDER.get(str(expected), 999)
    if actual_rank < expected_rank:
        failures.append(EvalFailure(check, expected, actual))


def check_expectations(
    *,
    expected: dict[str, Any],
    state: AgentState,
    selected_doc_ids: list[str],
    retrieved_doc_ids: list[str],
    selected_pages_: list[str],
    selected_sources: list[str],
    selected_source_sequence_: list[str],
    selected_text: str,
) -> list[EvalFailure]:
    failures: list[EvalFailure] = []
    generated_query = state["retrieval_query"]
    generated_query_text = normalize_text(generated_query)

    if "route" in expected and expected["route"] != state["route"]:
        failures.append(EvalFailure("route", expected["route"], state["route"]))

    if "route_confidence" in expected and expected["route_confidence"] != state["route_confidence"]:
        failures.append(
            EvalFailure("route_confidence", expected["route_confidence"], state["route_confidence"])
        )
    check_min_confidence(
        failures=failures,
        check="min_route_confidence",
        expected=expected.get("min_route_confidence"),
        actual=state["route_confidence"],
    )

    if "rewrite_used" in expected and expected["rewrite_used"] != state["rewrite_used"]:
        failures.append(
            EvalFailure("rewrite_used", expected["rewrite_used"], state["rewrite_used"])
        )
    if "rewrite_source" in expected and expected["rewrite_source"] != state["rewrite_source"]:
        failures.append(
            EvalFailure("rewrite_source", expected["rewrite_source"], state["rewrite_source"])
        )
    if "rewrite_confidence" in expected and expected["rewrite_confidence"] != state["rewrite_confidence"]:
        failures.append(
            EvalFailure("rewrite_confidence", expected["rewrite_confidence"], state["rewrite_confidence"])
        )
    check_min_confidence(
        failures=failures,
        check="min_rewrite_confidence",
        expected=expected.get("min_rewrite_confidence"),
        actual=state["rewrite_confidence"],
    )

    if "generated_query_exact" in expected and expected["generated_query_exact"] != generated_query:
        failures.append(
            EvalFailure("generated_query_exact", expected["generated_query_exact"], generated_query)
        )

    for term in expected.get("generated_query_terms_all", []):
        if normalize_text(term) not in generated_query_text:
            failures.append(EvalFailure("generated_query_terms_all", term, generated_query))

    query_terms_any = expected.get("generated_query_terms_any", [])
    if query_terms_any and not any(normalize_text(term) in generated_query_text for term in query_terms_any):
        failures.append(
            EvalFailure("generated_query_terms_any", query_terms_any, generated_query)
        )

    forbidden_query_hits = [
        term for term in expected.get("generated_query_forbidden_terms_any", [])
        if normalize_text(term) in generated_query_text
    ]
    if forbidden_query_hits:
        failures.append(
            EvalFailure("generated_query_forbidden_terms_any", [], forbidden_query_hits)
        )

    all_selected_docs = expected.get("selected_document_ids_all", [])
    for document_id in all_selected_docs:
        if document_id not in selected_doc_ids:
            failures.append(
                EvalFailure("selected_document_ids_all", document_id, selected_doc_ids)
            )

    any_selected_docs = expected.get("selected_document_ids_any", [])
    if any_selected_docs and not any(document_id in selected_doc_ids for document_id in any_selected_docs):
        failures.append(
            EvalFailure("selected_document_ids_any", any_selected_docs, selected_doc_ids)
        )

    all_retrieved_docs = expected.get("retrieved_document_ids_all", [])
    for document_id in all_retrieved_docs:
        if document_id not in retrieved_doc_ids:
            failures.append(
                EvalFailure("retrieved_document_ids_all", document_id, retrieved_doc_ids)
            )

    any_pages = [str(page) for page in expected.get("selected_pages_any", [])]
    if any_pages and not any(page in selected_pages_ for page in any_pages):
        failures.append(EvalFailure("selected_pages_any", any_pages, selected_pages_))

    all_pages = [str(page) for page in expected.get("selected_pages_all", [])]
    for page in all_pages:
        if page not in selected_pages_:
            failures.append(EvalFailure("selected_pages_all", page, selected_pages_))

    for term in expected.get("required_terms_all", []):
        if normalize_text(term) not in selected_text:
            failures.append(EvalFailure("required_terms_all", term, "not found"))

    for term in expected.get("selected_text_terms_all", []):
        if normalize_text(term) not in selected_text:
            failures.append(EvalFailure("selected_text_terms_all", term, "not found"))

    text_terms_any = expected.get("selected_text_terms_any", [])
    if text_terms_any and not any(normalize_text(term) in selected_text for term in text_terms_any):
        failures.append(EvalFailure("selected_text_terms_any", text_terms_any, "not found"))

    for index, group in enumerate(expected.get("selected_text_term_groups_all", []), start=1):
        if isinstance(group, dict):
            terms = group.get("terms", [])
            label = group.get("name") or f"group_{index}"
        else:
            terms = group
            label = f"group_{index}"
        if not isinstance(terms, list) or not terms:
            failures.append(EvalFailure("selected_text_term_groups_all", label, "invalid group"))
            continue
        if not any(normalize_text(term) in selected_text for term in terms):
            failures.append(EvalFailure("selected_text_term_groups_all", {label: terms}, "not found"))

    forbidden_hits = [
        term for term in expected.get("forbidden_terms_any", [])
        if normalize_text(term) in selected_text
    ]
    if forbidden_hits:
        failures.append(EvalFailure("forbidden_terms_any", [], forbidden_hits))

    selected_source_text = normalize_text(" ".join(selected_sources))
    forbidden_source_hits = [
        term
        for term in expected.get("selected_source_names_forbidden_any", [])
        if normalize_text(term) in selected_source_text
    ]
    if forbidden_source_hits:
        failures.append(
            EvalFailure("selected_source_names_forbidden_any", [], forbidden_source_hits)
        )

    if "max_foreign_source_count" in expected and selected_source_sequence_:
        primary_source = selected_source_sequence_[0]
        foreign_source_count = sum(
            1 for source in selected_source_sequence_ if source != primary_source
        )
        if foreign_source_count > int(expected["max_foreign_source_count"]):
            failures.append(
                EvalFailure(
                    "max_foreign_source_count",
                    expected["max_foreign_source_count"],
                    foreign_source_count,
                )
            )

    check_min_confidence(
        failures=failures,
        check="min_confidence",
        expected=expected.get("min_confidence"),
        actual=state["retrieval_confidence"],
    )

    if "confidence" in expected and expected["confidence"] != state["retrieval_confidence"]:
        failures.append(EvalFailure("confidence", expected["confidence"], state["retrieval_confidence"]))

    if "fallback_reason" in expected and expected["fallback_reason"] != state["fallback_reason"]:
        failures.append(
            EvalFailure("fallback_reason", expected["fallback_reason"], state["fallback_reason"])
        )

    return failures


def result_to_json(result: EvalResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "passed": result.passed,
        "failures": [
            {
                "check": failure.check,
                "expected": failure.expected,
                "actual": failure.actual,
            }
            for failure in result.failures
        ],
        **result.summary,
    }


def print_human_report(results: list[EvalResult]) -> None:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(f"Chat-to-retrieval evals: {passed}/{total} passed")
    print()
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name}")
        print(f"  user content: {result.summary.get('user_content')}")
        print(f"  generated query: {result.summary.get('retrieval_query')}")
        print(
            "  route: "
            f"{result.summary.get('route')} "
            f"({result.summary.get('route_confidence')})"
        )
        print(
            "  rewrite: "
            f"{result.summary.get('rewrite_source')} "
            f"({result.summary.get('rewrite_confidence')})"
        )
        print(f"  retrieval confidence: {result.summary.get('retrieval_confidence')}")
        print(f"  selected docs: {result.summary.get('selected_document_ids')}")
        print(f"  selected pages: {result.summary.get('selected_pages')}")
        if result.summary.get("fallback_reason"):
            print(f"  fallback: {result.summary['fallback_reason']}")
        for failure in result.failures:
            print(
                f"  - {failure.check}: expected {failure.expected!r}, "
                f"actual {failure.actual!r}"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run data-driven chat-to-retrieval evals."
    )
    parser.add_argument(
        "cases",
        nargs="?",
        default="evals/rag_retrieval_cases.json",
        help="Path to a JSON eval case file.",
    )
    parser.add_argument(
        "--user-id",
        help="Default user id for cases that do not specify user_id.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human report.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the full JSON report.",
    )
    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    defaults, cases = load_cases(cases_path)
    default_user_id = args.user_id or defaults.get("user_id")

    fixture_documents = defaults.get("_fixture_documents")
    db = None
    try:
        if fixture_documents:
            default_user_id = str(default_user_id or "fixture-user")
            document_context = FixtureDocumentContext(
                fixture_documents,
                default_user_id=default_user_id,
            )
            route_planner = ChatRoutePlanner(
                document_context.available_document_count,
                llm_factory=failing_llm_factory,
            )
        else:
            db = SessionLocal()
            rag_service = RagService(get_vector_db())
            document_context = ChatDocumentContext(db, rag_service=rag_service)
            route_planner = ChatRoutePlanner(document_context.available_document_count)
        results = [
            evaluate_case(
                case=case,
                defaults=defaults,
                route_planner=route_planner,
                document_context=document_context,
                default_user_id=default_user_id,
            )
            for case in cases
        ]
    finally:
        if db is not None:
            db.close()

    report = {
        "passed": sum(1 for result in results if result.passed),
        "total": len(results),
        "results": [result_to_json(result) for result in results],
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(results)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
