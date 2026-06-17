Here’s a practical TODO list you can use to start implementing the “general chatbot + optional uploaded-docs RAG” experience.

## Phase 1: Core behavior

### 1. Define chat modes

- [x] Add chat mode support:
  - `auto`
  - `general_only`
  - `documents_only`
- [x] Default new conversations to `auto`.
- [x] Add mode to the chat request payload.
- [x] Add mode to conversation/session state.
- [x] Expose mode selection in the UI.

```ts
type ChatMode = "auto" | "general_only" | "documents_only";
```

---

### 2. Add intent routing

- [ ] Create a router that classifies each message into:
  - `general_chat`
  - `document_question`
  - `summarize_document`
  - `needs_clarification`
- [ ] Include recent chat history in the router input.
- [ ] Include available uploaded document metadata in the router input.
- [ ] Make router return JSON.
- [ ] Add fallback behavior if router output is invalid.
- [ ] Log router decisions for debugging.

```ts
type ChatRoute =
  | "general_chat"
  | "document_question"
  | "summarize_document"
  | "needs_clarification";
```

---

### 3. Implement route decision logic

- [ ] If mode is `general_only`, skip document retrieval.
- [ ] If mode is `documents_only`, force document route or clarification.
- [ ] If mode is `auto`, use the router result.
- [ ] If no documents exist and route is `document_question`, ask user to upload a document.
- [ ] If user references “this file”, “the PDF”, “the contract”, etc., bias toward document route.

---

## Phase 2: Document-aware retrieval

### 4. Add query rewriting

- [ ] Create a query rewriting step before document retrieval.
- [ ] Use the latest user message.
- [ ] Include recent chat history.
- [ ] Include selected document names if available.
- [ ] Return a standalone retrieval query.
- [ ] Return `null` if the query is not document-related.
- [ ] Log original query and rewritten query.

Example:

```text
User: What about termination?

Rewritten query: What does the uploaded agreement say about termination clauses?
```

---

### 5. Improve retrieval strategy

- [ ] Increase initial vector retrieval count to around 20–40 chunks.
- [ ] Add metadata filters:
  - user ID
  - conversation ID
  - selected document IDs
- [ ] Add keyword search fallback if available.
- [ ] Add hybrid search if your vector store supports it.
- [ ] Deduplicate chunks by chunk ID/document ID/page.
- [ ] Keep retrieval isolated per user to avoid data leakage.

---

### 6. Add reranking

- [ ] Add a reranking step after initial retrieval.
- [ ] Rerank top 20–40 retrieved chunks.
- [ ] Keep top 5–8 chunks for final context.
- [ ] Store reranker scores.
- [ ] Add minimum reranker score threshold.
- [ ] If all scores are weak, trigger “not found / clarification” flow.

---

### 7. Add retrieval confidence handling

- [ ] Define confidence levels:
  - `high`
  - `medium`
  - `low`
- [ ] Use similarity score, reranker score, and number of good chunks.
- [ ] If confidence is high, answer with docs.
- [ ] If confidence is medium, answer cautiously with citations.
- [ ] If confidence is low, do not generate a doc-grounded answer.
- [ ] Return a helpful fallback message.

Example fallback:

```text
I searched the uploaded documents but couldn't find enough relevant information
to answer that confidently. Would you like me to answer based on general
knowledge instead?
```

---

## Phase 3: Answer generation

### 8. Add document-grounded answer prompt

- [ ] Create a separate prompt for document-based answers.
- [ ] Instruct the model to answer only from retrieved context.
- [ ] Instruct the model to cite sources.
- [ ] Instruct the model to say when the answer is not found.
- [ ] Prevent the model from inventing clauses, numbers, dates, or names.
- [ ] Include file name, page number, and chunk text in the context.

---

### 9. Add general chat prompt

- [ ] Create a separate prompt for normal chatbot behavior.
- [ ] Do not include document context for general chat.
- [ ] Allow the model to answer normally.
- [ ] Keep normal conversation history available.
- [ ] Avoid unnecessary document references.

---

### 10. Add citation support

- [ ] Include citation metadata in retrieved chunks:
  - document ID
  - file name
  - page number
  - section title
  - chunk ID
  - quoted text/snippet
- [ ] Return citations separately from answer text.
- [ ] Display citations in the UI.
- [ ] Make citations clickable if possible.
- [ ] Highlight or preview the cited snippet.

```ts
type Citation = {
  documentId: string;
  fileName: string;
  pageNumber?: number;
  sectionTitle?: string;
  chunkId: string;
  quote?: string;
};
```

---

## Phase 4: Upload experience

### 11. Improve document upload handling

- [ ] Show upload progress.
- [ ] Show indexing/processing progress.
- [ ] Prevent asking document questions until indexing completes.
- [ ] Store file metadata.
- [ ] Extract page numbers where possible.
- [ ] Extract headings/sections where possible.
- [ ] Split documents into chunks.
- [ ] Store chunks with metadata.
- [ ] Generate embeddings.
- [ ] Save chunks to vector store.

---

### 12. Generate upload summary

After upload/indexing:

- [ ] Generate a short summary of the uploaded document.
- [ ] Detect document type if possible:
  - contract
  - policy
  - invoice
  - report
  - resume
  - specification
  - unknown
- [ ] Extract key topics.
- [ ] Generate 3–5 suggested questions.
- [ ] Show this summary in the chat.

Example:

```text
Uploaded: Vendor Agreement.pdf

I found:
- Type: Agreement
- Main topics: payment terms, confidentiality, termination, liability
- Length: 12 pages

You can ask:
- What is the termination period?
- Are there any payment deadlines?
- What are the liability limits?
```

---

### 13. Add document selection

- [ ] Allow users to select which documents should be used.
- [ ] Add “use all documents” option.
- [ ] Add “ignore documents” option.
- [ ] Pass selected document IDs to retrieval.
- [ ] Show which documents are active for the current chat.

---

## Phase 5: UI quick wins

### 14. Show document usage status

- [ ] Show “Searching uploaded documents...” while retrieval is running.
- [ ] Show “Answering from uploaded documents” for RAG answers.
- [ ] Show “Answering generally” for non-RAG answers.
- [ ] Show “No relevant document content found” when retrieval fails.
- [ ] Show selected chat mode in the composer.

---

### 15. Add suggested actions

After document upload:

- [ ] Add `Summarize`
- [ ] Add `Find risks`
- [ ] Add `Extract action items`
- [ ] Add `Find dates and deadlines`
- [ ] Add `Ask a question`

After a document answer:

- [ ] Add 2–3 relevant follow-up questions.
- [ ] Let users click a suggestion to send it as a message.

---

### 16. Add clarification flow

- [ ] If the user says “summarize it” and multiple documents exist, ask which one.
- [ ] If the user says “what about this?” and intent is unclear, ask a follow-up.
- [ ] Ask only one focused clarification question.
- [ ] Do not ask clarification if the answer is obvious from history.

Examples:

```text
Which uploaded document should I summarize?
```

```text
Do you want me to answer from the uploaded documents or generally?
```

---

## Phase 6: Observability and evaluation

### 17. Add logs

Log the following per chat request:

- [ ] user ID
- [ ] conversation ID
- [ ] chat mode
- [ ] user message
- [ ] detected route
- [ ] router confidence/reason
- [ ] rewritten retrieval query
- [ ] selected document IDs
- [ ] retrieved chunk IDs
- [ ] similarity scores
- [ ] reranker scores
- [ ] final chunks used
- [ ] whether documents were used
- [ ] citations returned
- [ ] fallback triggered or not
- [ ] latency per step

---

### 18. Add basic analytics

Track:

- [ ] percentage of questions routed to docs
- [ ] percentage of retrieval failures
- [ ] percentage of answers with citations
- [ ] average retrieval confidence
- [ ] average response latency
- [ ] most queried documents
- [ ] most common fallback cases

---

### 19. Create an evaluation set

- [ ] Collect 20–50 real uploaded documents or test documents.
- [ ] Create sample questions for each document.
- [ ] Define expected answer or expected source chunk.
- [ ] Test:
  - router accuracy
  - retrieval accuracy
  - citation correctness
  - final answer quality
- [ ] Run evals before changing retrieval logic.

Example:

```ts
type RagEvalCase = {
  documentId: string;
  question: string;
  expectedAnswer?: string;
  expectedChunkIds?: string[];
  expectedCitationPages?: number[];
};
```

---

## Phase 7: Safety and data isolation

### 20. Add access control checks

- [ ] Ensure retrieval filters by `userId` or tenant ID.
- [ ] Ensure users cannot retrieve chunks from other users.
- [ ] Validate selected document IDs belong to the requesting user.
- [ ] Do not expose internal chunk IDs if not needed.
- [ ] Redact sensitive logs if required.
- [ ] Add rate limits for upload and chat requests.

---

### 21. Add document-only safety behavior

For `documents_only` mode:

- [ ] If answer is not in docs, say it is not in docs.
- [ ] Do not answer using general knowledge.
- [ ] Do not guess missing numbers/dates/names.
- [ ] Always include citations when answering.
- [ ] If no citations, do not produce a document answer.

---

# Suggested implementation order

If you want to start tomorrow, I’d do it in this order:

```text
1. Add ChatMode: auto / general_only / documents_only
2. Add intent router
3. Add query rewriting
4. Increase retrieval count
5. Add citation metadata and return citations
6. Add low-confidence fallback
7. Add document-grounded prompt
8. Add upload summary + suggested questions
9. Add reranking
10. Add hybrid search
11. Add observability logs
12. Add eval set
```

---

# Minimal first milestone

Your first milestone can be:

```text
A user can upload a document, ask a question, and the chatbot decides whether
to answer generally or from the document. If it uses the document, it provides
citations. If it cannot find the answer, it says so clearly.
```

That alone will noticeably improve the chatbot experience.
