# Chat-to-Retrieval Evals

These evals test the same query-generation path used by chat before vector
retrieval. Each case starts with optional chat history plus a latest user
message. The runner generates a standalone `retrieval_query`, sends that query
to the vector store, and checks both the generated query and selected context.

Run from the backend root:

```powershell
python scripts/run_rag_evals.py evals/rag_retrieval_cases.json
```

Useful options:

```powershell
python scripts/run_rag_evals.py evals/rag_retrieval_cases.json --user-id <user-id>
python scripts/run_rag_evals.py evals/rag_retrieval_cases.json --json
python scripts/run_rag_evals.py evals/rag_retrieval_cases.json --output evals/latest-report.json
```

Case format:

```json
{
  "name": "tata_followup_resource_allocation",
  "chat_mode": "document_only",
  "history": [
    {"role": "user", "content": "Summarize the Tata Sons SOW."},
    {"role": "assistant", "content": "It is about Deep Research AI RAG."}
  ],
  "user_content": "What is the resource allocation?",
  "expected": {
    "route": "document_question",
    "generated_query_terms_all": ["Tata Sons", "resource allocation"],
    "generated_query_forbidden_terms_any": ["Nippon", "Wipro"],
    "min_rewrite_confidence": "medium",
    "selected_document_ids_any": ["f814d17c-d0f8-4b5c-9cfd-907314ec2b15"],
    "selected_pages_any": ["12"],
    "required_terms_all": ["Lead Gen AI Engineer", "Data Engineer"]
  }
}
```

Required fields:

- `user_content`: latest user message to evaluate.
- `expected`: expectation object.

Optional fields:

- `chat_mode`: `auto`, `document_only`, or `general_only`. Defaults to
  `defaults.chat_mode`, then `auto`.
- `history`: array of `{ "role": "user" | "assistant", "content": "..." }`.
- `user_id`, `message_id`, `conversation_id`, `model_id`: override defaults for
  a single case.

Query-generation expectation fields:

- `route`: exact expected route.
- `route_confidence`: exact route confidence.
- `min_route_confidence`: minimum route confidence: `low`, `medium`, or `high`.
- `rewrite_used`: exact boolean.
- `rewrite_source`: exact source: `llm`, `heuristic`, or `none`.
- `rewrite_confidence`: exact rewrite confidence.
- `min_rewrite_confidence`: minimum rewrite confidence.
- `generated_query_exact`: exact generated retrieval query.
- `generated_query_terms_all`: every listed term must appear in the generated
  retrieval query.
- `generated_query_terms_any`: at least one listed term must appear in the
  generated retrieval query.
- `generated_query_forbidden_terms_any`: none of these terms may appear in the
  generated retrieval query.

Retrieval expectation fields:

- `selected_document_ids_any`: at least one listed document must be selected.
- `selected_document_ids_all`: every listed document must be selected.
- `retrieved_document_ids_all`: every listed document must appear in initial retrieval.
- `selected_pages_any`: at least one listed page must be selected.
- `selected_pages_all`: every listed page must be selected.
- `required_terms_all`: every listed term must appear in selected context text.
- `forbidden_terms_any`: none of these terms may appear in selected context text.
- `min_confidence`: actual retrieval confidence must be at least `low`,
  `medium`, or `high`.
- `confidence`: exact expected retrieval confidence.
- `fallback_reason`: exact fallback reason, or `null` when no fallback is expected.
