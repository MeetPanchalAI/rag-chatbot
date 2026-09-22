# RAG Document Chatbot

Ask questions about a PDF and get answers grounded in it, with the page and
section they came from. When the document does not hold the answer, it says so
instead of guessing.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # then put your OpenAI key in it
```

## Run

```bash
python -m app.ingest_cli corpus/free221.pdf corpus/Lecture10.pdf
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** for the UI, or `/docs` for the API.

`corpus/` holds the two PDFs the evaluation is built on. Any PDF works — upload
one in the UI, or pass it to the same command.

## The UI

One page at `/`:

- **Upload a PDF**, or remove one you are done with.
- **Ask questions**, with follow-ups. Citations appear under each answer, and a
  refusal looks different from an answer.
- **Several conversations**, stored server-side, so they survive a closed browser
  with their citations and retrieval traces intact.
- **Show what was retrieved** puts every retrieved chunk, its score and whether
  it became evidence under the reply.
- **Run the evaluation**, label it with what you changed, and open any past run.
- **Activity** counts the last 500 questions: answered against refused, median
  latency, median retrieval score, and which guards fired.

It is one HTML file with no build step and no dependencies. The brief asks for a
lightweight UI, so this is a test harness with a clean face.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The UI |
| `GET /health` | Status and how much is indexed |
| `GET /documents` | What has been ingested |
| `POST /ingest` | Upload a PDF |
| `DELETE /documents/{id}` | Remove a document, its chunks and its stored PDF |
| `POST /chat` | Ask a question |
| `GET /activity` | Dashboard counts and recent questions |
| `GET`/`POST /conversations` | List, or start one |
| `GET`/`DELETE /conversations/{id}` | Read one back, or remove it |
| `POST /eval/run` | Start an evaluation |
| `GET /eval/status` | Progress and results of the latest run |
| `GET /eval/runs` | Every past run |
| `GET`/`DELETE /eval/runs/{id}` | Read one in full, or remove it |

### Asking a question

```bash
curl -X POST localhost:8000/chat -H "content-type: application/json" -d '{
  "question": "What is centripetal acceleration?"
}'
```

```json
{
  "answer": "Its magnitude is v squared over r, directed toward the centre.",
  "answerable": true,
  "citations": [
    {
      "document": "Lecture10",
      "page_start": 2,
      "page_end": 2,
      "section": "Centripetal acceleration",
      "display": "Page 2 - \"Centripetal acceleration\""
    }
  ]
}
```

Optional fields: `doc_id` to search one document, `debug: true` to see what was
retrieved, and for follow-ups either `history` (nothing is stored) or
`conversation_id` (the server keeps the history and records both turns).

## Tests

```bash
pytest
```

154 tests, all offline. The embedding model and the LLM are replaced by fakes and
test PDFs are generated in memory, so no key and no network are needed.

## Evaluation

20 questions over the two PDFs in `corpus/`, four in each of the five behaviours
the brief names. Five metrics per run, overall and by category:

| Metric | Answers |
| --- | --- |
| Retrieval | Did retrieval find the evidence? |
| Correctness | Is the answer right? |
| Groundedness | Is it supported by the evidence shown? |
| Citation support | Does the cited page hold the claim? |
| Abstention | Does it refuse when the document cannot answer? |

```bash
python -m eval.run_eval                 # or the Evaluation tab
python -m eval.run_eval --no-judge      # retrieval only, no judge calls
```

Every run is stored with the settings that produced it, so two runs can be
compared. Latest results: **[eval/results.md](eval/results.md)**. Method and
limits: **[EVAL.md](EVAL.md)**.

## Configuration

Everything comes from the environment; see `.env.example`. No secrets in code.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. The server refuses to start without it |
| `LLM_MODEL` | `gpt-5.6-luna` | Writes the answer |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeds chunks and queries |
| `REWRITE_MODEL` | same as `LLM_MODEL` | Rewrites follow-up questions |
| `JUDGE_MODEL` | same as `LLM_MODEL` | Grades answers during evaluation |
| `RERANK_MODEL` | same as `LLM_MODEL` | Reorders retrieved candidates |
| `LLM_TEMPERATURE` | `1.0` | gpt-5.6-luna is a reasoning model and takes only its default |
| `JUDGE_TEMPERATURE` | same as `LLM_TEMPERATURE` | Temperature for the judge |
| `LLM_REASONING_EFFORT` | `low` | Blank omits the parameter entirely |
| `CHUNK_TOKENS` | `500` | Target chunk size |
| `RETRIEVER_TOP_K` | `6` | Chunks retrieved per question |
| `HYBRID_SEARCH` | `true` | Fuse BM25 keyword search with the dense search |
| `RERANK` | `false` | Let the model reorder candidates. One extra call per question |
| `RERANK_CANDIDATES` | `20` | How many candidates the reranker sees |
| `MAX_CONTEXT_TOKENS` | `4000` | Ceiling on evidence sent to the model |

Works with any OpenAI-compatible endpoint via `OPENAI_BASE_URL`.

## Notes

- **Large PDFs**: use `python -m app.ingest_cli`. A long document is thousands of
  chunks and several minutes of embedding calls, which outlasts an HTTP timeout.
  The upload endpoint is capped by `MAX_UPLOAD_MB`.
- **Scanned PDFs** are rejected with a clear message. There is no OCR.
- **Page numbers** are PDF positions, not the numbers printed on the page. They
  differ in a document with front matter.
- **Storage** is one SQLite file at `DATA_DIR/app.db`, with the original PDFs in
  `DATA_DIR/documents/` so a corpus can be re-chunked later. Nothing in
  `DATA_DIR` is committed.

Design and trade-offs: **[DESIGN.md](DESIGN.md)**. AI tooling:
**[AI_USAGE.md](AI_USAGE.md)**.
