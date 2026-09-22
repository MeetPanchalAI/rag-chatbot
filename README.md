# RAG Document Chatbot

Ask questions about any PDF and get answers grounded in it, with page and
section citations. When the document does not answer the question, the system
says so instead of guessing.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # then put your API key in it
```

## Run

```bash
# 1. Index a PDF (use this for large files rather than the endpoint)
python -m app.ingest_cli path/to/document.pdf

# 2. Start the API
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000** for the UI, or `/docs` for the API.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The UI |
| `GET /health` | Service status and how much is indexed |
| `GET /documents` | What has been ingested, with document ids |
| `POST /ingest` | Upload a PDF (multipart). Small files only; see the note below |
| `DELETE /documents/{id}` | Remove a document, its chunks and its stored PDF |
| `POST /chat` | Ask a question |
| `GET`/`POST /conversations` | List or start a stored conversation |
| `GET`/`DELETE /conversations/{id}` | Read one back, or remove it |
| `POST /eval/run` | Start an evaluation run |
| `GET /eval/status` | Progress and results of the latest run |
| `GET /eval/runs` | Every past run, newest first |
| `GET`/`DELETE /eval/runs/{id}` | Read a past run in full, or remove it |

## The UI

One page at `/`, enough to exercise everything without a REST client:

- **Upload a PDF** and watch the page and chunk counts change.
- **Ask questions**, with follow-ups; citations appear under each answer and a
  refusal is styled differently from an answer.
- **Keep several conversations** side by side. They are stored server side, so
  they survive a cleared browser and come back with their citations and
  retrieval traces intact.
- **Remove a document** you are done with, chunks and stored PDF together.
- **Label an evaluation run** with what you changed, and click any past run to
  read it back.
- **Show what was retrieved** puts every retrieved chunk, its score, and which
  ones became evidence under the reply.
- **Scope the search** to one document or leave it across all of them.
- **Run the evaluation** from the Evaluation tab, with a progress bar and the
  full summary and per-question table when it finishes.

It is one self-contained HTML file with no build step, no package manager and
no CDN. The brief says not to spend significant time on frontend work, so this
is a test harness with a clean face, not a product.

### Asking a question

```bash
curl -X POST localhost:8000/chat -H "content-type: application/json" -d '{
  "question": "What is the maximum amount allowed?"
}'
```

```json
{
  "answer": "The maximum amount allowed is 50 units.",
  "answerable": true,
  "citations": [
    {
      "document": "Handbook",
      "page_start": 47,
      "page_end": 47,
      "section": "Eligibility Requirements",
      "display": "Page 47 - \"Eligibility Requirements\""
    }
  ]
}
```

Optional fields: `doc_id` to search one document instead of all of them,
`debug: true` to see what was retrieved, and for follow-ups either `history`
(stateless, nothing is stored) or `conversation_id` (the server supplies the
history and records both turns).

### Follow-up questions

Send the previous turns in `history`. A follow-up like *"What about
international applicants?"* is rewritten into a standalone query before the
search, because the question alone retrieves nothing useful.

```json
{
  "question": "What about international applicants?",
  "history": [
    {"role": "user", "content": "What are the eligibility requirements?"},
    {"role": "assistant", "content": "Applicants must be at least eighteen..."}
  ]
}
```

### Seeing what was retrieved

`"debug": true` adds a `trace` to the response: the rewritten query, every
retrieved chunk with its score, which ones became evidence, and which guards
fired. The same information is written to the log for every request.

## Tests

```bash
pytest
```

Every test runs offline. The embedding model and the LLM are replaced by
fakes, and test PDFs are generated in memory, so there is no API key needed and
no network call.

## Evaluation

`eval/questions.jsonl` holds the question set, one JSON object per line:

| Field | Meaning |
| --- | --- |
| `id`, `type` | Identifier and category |
| `question` | What to ask |
| `history` | Previous turns, for follow-up questions |
| `answerable` | Whether the document should be able to answer it |
| `gold_pages` | Pages that hold the answer |
| `expected_answer_contains` | Words the answer should contain |

The set in the repository is 18 questions written against this assignment
brief, in the five categories it asks for:

| Category | Count | What it tests |
| --- | --- | --- |
| `factual` | 4 | One passage, one page |
| `multi_passage` | 4 | Answers that span two pages |
| `follow_up` | 3 | Questions meaningless without the history |
| `unanswerable` | 3 | Does it admit it does not know? |
| `similar_sections` | 4 | Several sections compete for the same query |

Every gold page was checked against the indexed text, so a retrieval miss is a
real miss and not a bad label.

The PDF itself is not committed. To reproduce the run, index your own copy of
the assignment brief first:

```bash
python -m app.ingest_cli AI_Engineering_Task_1_RAG.pdf
python -m eval.run_eval
```

This writes `eval/results.md` (a report) and `eval/results.jsonl` (raw scores).
It needs an indexed document and an API key, and costs one or two model calls
per question. See DESIGN.md for what the metrics mean.

To evaluate a different PDF, replace `questions.jsonl` with questions written
against it. Nothing in the code is tied to this document.

## Configuration

Everything is set through the environment; see `.env.example`. No secrets in
code. The settings you are most likely to change:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-5.6-luna` | Model that writes the answer |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model that embeds chunks and queries |
| `LLM_TEMPERATURE` | `1.0` | gpt-5.6-luna is a reasoning model and only accepts its default |
| `LLM_REASONING_EFFORT` | `low` | Leave blank to omit the parameter entirely |
| `CHUNK_TOKENS` | `500` | Target chunk size |
| `RETRIEVER_TOP_K` | `6` | Chunks retrieved per question |
| `MAX_CONTEXT_TOKENS` | `4000` | Ceiling on evidence sent to the model |

The API works with any OpenAI-compatible endpoint via `OPENAI_BASE_URL`.

## Notes and limits

- **Large PDFs**: use `python -m app.ingest_cli`. A long document is thousands
  of chunks and several minutes of embedding calls, which outlasts a normal
  HTTP timeout. The upload endpoint is capped by `MAX_UPLOAD_MB`.
- **Scanned PDFs** are rejected with a clear error. There is no OCR.
- **Page numbers** are PDF page positions, not the numbers printed on the page.
  In a document with front matter the two differ.
- **Storage** is one SQLite file, `DATA_DIR/app.db`, holding documents, chunks
  with their vectors, conversations and evaluation runs. Original PDFs sit in
  `DATA_DIR/documents/` so a corpus can be re-chunked later without being
  re-supplied; set `KEEP_SOURCE_PDF=false` to skip that. Nothing in `DATA_DIR`
  is committed.
- **An older index** written as `chunks.jsonl` + `embeddings.npy` is imported
  automatically on first start, and the old files are renamed `*.migrated`.

Full design rationale, trade-offs and limitations: [DESIGN.md](DESIGN.md).
