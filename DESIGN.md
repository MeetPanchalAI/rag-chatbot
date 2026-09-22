# Design

## Shape of the system

```
PDF ──► ingestion ──► chunks + page/section metadata ──► embeddings ──► index
                                                                          │
question + history ──► rewrite ──► retrieve ◄─────────────────────────────┘
                                      │
                                      ▼
                           numbered evidence ──► LLM ──► guards ──► answer + citations
```

| Module | Responsibility |
| --- | --- |
| `ingestion.py` | PDF to chunks. Knows nothing about embeddings or models |
| `vector_store.py` | Storage and cosine search |
| `retrieval.py` | Query rewriting, search, evidence assembly |
| `generation.py` | Prompt, and the guards on what the model returns |
| `pipeline.py` | Wires the flow together; used by both the API and the eval |
| `providers.py` | The only place that talks to OpenAI |
| `main.py` | HTTP: validation, wiring, error mapping |

Dependencies point one way: `main` → `pipeline` → `retrieval`/`generation` →
`vector_store`/`ingestion` → `providers`/`schemas`/`config`.

## Chunking

Roughly 500 tokens with 75 tokens of overlap, split on section boundaries
first, then paragraphs, then sentences, then words.

500 tokens is about two paragraphs: large enough that a passage still makes
sense on its own, small enough that one retrieved chunk is mostly answer rather
than surrounding material. The overlap keeps a sentence that straddles a
boundary retrievable from either side.

Sections are found in this order, strongest signal first:

1. **The table of contents the PDF ships with.** Authoritative titles, and no
   guessing.
2. **Numbered headings** such as `3.2 Eligibility`.
3. **Font size and weight** compared with the median body text.

If fewer than two headings turn up, or more than 30% of lines look like
headings, the result is treated as noise and discarded. Citations then carry
page numbers only. A missing section label is honest; an invented one is not.

Two further details matter more than they look:

- **Column-aware extraction.** Reading a two-column page top to bottom
  interleaves the columns into unusable text. Words are assigned to a column
  before being grouped into lines. A page only counts as two columns if it has
  a genuinely empty strip down the middle: a centred title crosses that strip,
  and splitting the page in half would cut the title in two.
- **Fragment merging.** Heading detection is deliberately generous, so a styled
  page can produce "sections" one line long. Those are folded back into the
  section above, heading line included, rather than becoming chunks too short
  to retrieve on.
- **Enriched embedding text.** A chunk from the middle of a section often reads
  as *"this gives us the result above"*, which embeds to nothing useful. The
  document and section title are prefixed to the text we embed. The text we
  display and cite is untouched.

## Retrieval

Dense cosine similarity over `text-embedding-3-small`, top 6 chunks, trimmed to
a 4000-token evidence budget.

Dense-only is the right starting point: it handles paraphrase, which is how
people actually ask questions, and it is one moving part instead of two. The
known weakness is exact identifiers — codes, names, amounts — where a keyword
index does better. Hybrid search is the first thing to add if the evaluation
shows those misses, and the evaluation is designed to show them.

**Follow-up questions** are rewritten into standalone queries before the search.
*"What about international applicants?"* retrieves nothing on its own. The
rewrite costs one small model call, and only happens when there is history. A
rewrite that comes back empty or rambling is discarded in favour of the original
question. History is used to interpret the question and never enters the
evidence.

## Embeddings and storage

`text-embedding-3-small` keeps embedding and generation with one provider and
one key.

Everything durable lives in one SQLite file, `data/app.db`: documents, chunks
with their vectors, conversations, and evaluation runs with their results. The
original PDFs sit beside it in `data/documents/`.

**Why the vectors are in the database and the PDFs are not.** A PDF is only ever
read whole, so a blob in a table makes every scan and backup carry weight for
nothing; it belongs on disk. A vector is the opposite. Chunks and embeddings
used to be two files kept in step by line order, with nothing enforcing it —
which is precisely why a document could not be deleted. In one row they are
written in the same transaction and deleted together.

At load the vectors are stacked into a single numpy matrix, so search is what it
always was: one matrix multiply, well under a millisecond at tens of thousands
of chunks and far smaller than the embedding call before it. SQLite is the
durable copy; the matrix is a cache of it.

A hosted vector database earns its place when the corpus outgrows memory or
needs concurrent writers. Neither is true here, and SQLite costs nothing to run.

| Kept | Where | Why |
| --- | --- | --- |
| Documents, chunks, vectors | `app.db` | One transaction, deletable together |
| Original PDFs | `data/documents/` | Read whole; would bloat every query |
| Conversations and messages | `app.db` | Survive a cleared browser, with citations and traces |
| Evaluation runs and results | `app.db` | Comparing runs is the point of running them |
| Reports | `eval/results.md`, `.jsonl` | Readable artefacts of the latest run |

**This is beyond the brief.** The brief asks for a searchable representation and
says nothing about persistence. It was added because run-to-run comparison is
what makes evaluation useful, and because a corpus you cannot re-chunk or delete
from is awkward to iterate on.

## Citations

The model never writes a page number.

Evidence reaches it as a numbered list. It returns positions in that list. We
resolve those positions against the chunks we retrieved, and build the citation
from chunk metadata captured at ingestion time. A page number the model invents
has no route to the user.

Three guards enforce this:

| Guard | Behaviour |
| --- | --- |
| Out-of-range citation | Dropped, and logged |
| Answer claimed grounded with no valid citation | Downgraded to a refusal |
| Unparseable output | Retried once, then a refusal |

`test_invariants.py` checks both load-bearing properties end to end: the model
only ever sees retrieved chunks, and every citation resolves to a chunk that
was actually retrieved.

## What running the parser over a real PDF changed

The synthetic PDFs in the test suite are clean, and a real one was not. Running
the parser over the assignment brief itself found three defects that no
generated fixture would have shown:

| Defect | Effect | Fix |
| --- | --- | --- |
| Column split fired on a styled title page | The centred title was cut in half and read out of order | A two-column reading now needs an empty gutter |
| Heading rules caught body text | 38 "headings" in 129 lines, most of them wrapped sentences | Headings must start like titles, and bullets are excluded |
| One-line sections became chunks | Chunks of 28 characters, too short to retrieve on | Fragments merge into the section above |

Afterwards: 22 headings, matching the document's real section structure, and a
smallest chunk of 77 characters instead of 28. Each fix has a regression test.

The limitation this exposed and did not fix: a table row is read as one line of
run-together text. Tables lose their structure.

## Questions the document cannot answer

Two paths lead to a refusal:

1. **Nothing retrieved.** The model is not called at all; there is nothing to
   ground an answer in.
2. **The model says the evidence is insufficient**, by returning
   `answerable: false`.

There is no similarity threshold, on purpose. Cosine scores are poorly
calibrated across models and documents, and a fixed cutoff turns into false
refusals on valid questions. The model sees the evidence and is better placed
to judge.

That is a decision, not a certainty, so it is measured. Every request logs its
top retrieval score, and the evaluation reports the score distribution for
answerable and unanswerable questions separately. If they separate cleanly, a
threshold is a cheap extra safeguard and the data will say so.

## Failure handling

| Failure | Response |
| --- | --- |
| Empty, over-long, or malformed request | 422 |
| Unknown `doc_id` | 404 |
| Not a PDF | 415 |
| Upload over the size limit | 413 |
| Corrupt, encrypted, or text-free PDF | 422 with a specific message |
| Embedding or LLM failure | Retried once, then 502 |
| Provider timeout | 504 |
| Unparseable model output | Retried once, then a 200 refusal |
| Nothing retrieved | 200 refusal |

A failure to answer is a valid answer and returns 200. Only a broken request or
a broken dependency is an error status.

## Observability

Every `/chat` request logs one JSON line: request id, rewritten query, how much
was retrieved, top score, whether it was answerable, how many citations, which
guards fired, and latency. `"debug": true` returns the same detail in the
response, including per-chunk scores and the raw model output.

## The UI

A single static HTML file served at `/`, with no build step, no package manager
and no CDN. It exists to exercise ingestion, chat, follow-ups, the retrieval
trace and the evaluation without a REST client. The brief asks for a lightweight
UI and warns against spending real time on the frontend, so it stays a test
harness with a clean face.

Running the evaluation from a browser needs more than a plain request: eighteen
questions at one or two model calls each is about a minute, which is too long to
hold a connection open. `POST /eval/run` starts a worker thread and returns
immediately; the page polls `GET /eval/status` for progress and results. A
failure on that thread is captured into the run status rather than disappearing
into a log, and a second run is refused with a 409 while one is in flight.

Scoring lives in `app/evaluation.py`, not in the eval script, so the command
line and the UI run exactly the same code.

## Evaluation

20 questions over the two PDFs in `corpus/`, scored on five metrics and broken
down by category. Retrieval is scored arithmetically against gold pages; the answer is
scored by an LLM judge on four separate axes, never blended into one number.
Every run is stored with the settings that produced it.

Full methodology and its limits: **[EVAL.md](EVAL.md)**.

## Limitations, and what comes next

- **Mathematics and formulas** extract badly from PDFs. Superscripts flatten,
  symbols are lost. Answers quoting equations will be unreliable. A
  maths-aware extractor would be needed.
- **Tables and figures** are read as loose text. Their structure is lost.
- **No OCR**, so scanned documents are rejected rather than mishandled.
- **Page numbers are PDF positions**, not the numbers printed on the page.
  These differ in any document with front matter.
- **Parsing is pure Python** and therefore slow: minutes for a long book.
  Ingestion is a one-off cost, but it is not fast.
- **Single writer.** SQLite runs in WAL mode with one connection per thread and
  serialised writes, which is fine for one process. Several servers sharing a
  data directory would still contend.
- **No authentication.** Conversations are not scoped to a user, because there
  are no users. Anyone reaching the API sees all of them.
- **The keyword screen cannot judge answer quality.** An LLM judge would scale
  this, at the cost of its own bias.

In priority order, next: hybrid search for exact identifiers, a reranker for
documents with many near-identical sections, and an LLM judge for answer and
citation quality.

## Decisions at a glance

| Decision | Choice | Why | Cost |
| --- | --- | --- | --- |
| Retrieval | Dense only | Handles paraphrase; one moving part | Misses exact identifiers |
| Chunk size | 500 tokens, 75 overlap | Coherent passages, precise retrieval | Boundaries can still split an answer |
| Headings | Table of contents first | Authoritative, no guessing | Falls back to page-only citations |
| Embedded text | Prefixed with section title | Anchors context-free chunks | Slightly more tokens to embed |
| Storage | SQLite + numpy in memory | One transaction, no service to run | Single writer, not larger than memory |
| Citations | Evidence positions, resolved by us | Invented pages cannot reach the user | Needs guard logic |
| Abstention | Model's judgement, no threshold | Cosine scores are poorly calibrated | Measured rather than assumed |
| Follow-ups | Rewrite only when history exists | Follow-ups fail without it | One extra model call |
| Testing | Offline fakes | Fast, free, deterministic | Does not exercise the real provider |

## One deviation worth recording

The plan called for PyMuPDF. The Windows Application Control policy on the
development machine blocks its native library, so parsing uses **pdfplumber**
(text and layout) and **pypdf** (table of contents) instead. Both are pure
Python, so there is nothing to block.

Only the parsing section of `ingestion.py` changed. Everything below it works
on a list of `Line` objects and did not have to be touched, which is the
separation working as intended. The cost is speed: pdfplumber is noticeably
slower than PyMuPDF on long documents.
