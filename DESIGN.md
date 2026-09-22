# Design

## How it fits together

```
PDF ─► ingestion ─► chunks + page/section ─► embeddings ─► SQLite
                                                              │
question + history ─► rewrite ─► retrieve ◄───────────────────┘
                                    │
                          numbered evidence ─► LLM ─► guards ─► answer + citations
```

| Module | Does |
| --- | --- |
| `ingestion.py` | PDF to chunks. Knows nothing about embeddings or models |
| `vector_store.py` | Storage and search |
| `retrieval.py` | Query rewriting, search, building the evidence |
| `generation.py` | The prompt, and the guards on what comes back |
| `judge.py` | Scores answers during evaluation |
| `pipeline.py` | Wires the flow; used by the API and the evaluation alike |
| `providers.py` | The only file that talks to OpenAI |
| `main.py` | HTTP: validation, wiring, error mapping |

Imports point one way: `main` → `pipeline` → `retrieval`/`generation` →
`vector_store`/`ingestion` → `providers`/`schemas`/`config`.

## Chunking

About 500 tokens with 75 tokens of overlap, split on section boundaries first,
then paragraphs, then sentences, then words.

500 tokens is roughly two paragraphs — big enough that a passage still makes
sense alone, small enough that a retrieved chunk is mostly answer rather than
surrounding text. The overlap keeps a sentence that straddles a boundary
findable from either side.

Sections are found in this order:

1. **The table of contents the PDF ships with.** Real titles, no guessing.
2. **Numbered headings** like `3.2 Eligibility`.
3. **Font size and weight** against the median body text.

If fewer than two headings turn up, or more than 30% of lines look like
headings, the result is treated as noise and thrown away. Citations then carry
page numbers only. A missing section label is honest; an invented one is not.

Three details that matter more than they look:

- **Columns.** Reading a two-column page top to bottom interleaves the columns
  into nonsense. Words are assigned to a column first. A page only counts as two
  columns if it has an empty strip down the middle — a centred title crosses
  that strip, and splitting the page in half would cut the title in two.
- **Fragments.** Heading detection is generous, so a styled page can produce
  "sections" one line long. Those fold back into the section above rather than
  becoming chunks too short to retrieve on.
- **What we embed is not what we show.** A chunk from mid-section often reads as
  "this gives us the result above", which embeds to nothing useful. The document
  and section title are prefixed to the embedded text. The displayed text is
  untouched.

## Retrieval

Dense cosine similarity over `text-embedding-3-small`, top 6 chunks, trimmed to
a 4000-token evidence budget.

Dense-only is the right start: it handles paraphrase, which is how people ask
questions, and it is one moving part instead of two. Its known weakness is exact
identifiers — codes, names, amounts — where keyword search does better. Hybrid
search is the first thing to add, and the evaluation is built to show whether
it is needed.

**Follow-ups** are rewritten into standalone queries before the search. *"What
about international applicants?"* finds nothing on its own. The rewrite costs one
small model call and only runs when there is history. A rewrite that comes back
empty or rambling is thrown away in favour of the original question. History is
used to read the question, never as evidence.

## Storage

One SQLite file, `data/app.db`: documents, chunks with their vectors,
conversations, and evaluation runs. The original PDFs sit in `data/documents/`.

**Why vectors go in the database and PDFs do not.** A PDF is only ever read
whole, so putting it in a table you query makes every scan and backup carry
weight for nothing. A vector is the opposite. Chunks and embeddings used to be
two files kept in step by line order, with nothing enforcing it — which is
exactly why deleting a document was impossible. In one row they are written in
one transaction and deleted together.

At load the vectors are stacked into one numpy matrix, so search is a single
matrix multiply: well under a millisecond at tens of thousands of chunks, far
smaller than the embedding call before it. SQLite is the durable copy; the matrix
is a cache of it.

A hosted vector database earns its place when the corpus outgrows memory or
needs concurrent writers. Neither is true here.

## Citations

**The model never writes a page number.**

Evidence reaches it as a numbered list. It returns positions in that list. We
resolve those positions against the chunks we retrieved and build the citation
from metadata captured at ingestion. A page number the model invents has no route
to the user.

| Guard | What happens |
| --- | --- |
| Citation outside the evidence list | Dropped and logged |
| Claimed grounded, no valid citation | Downgraded to a refusal |
| Unparseable output | Retried once, then a refusal |

`tests/test_invariants.py` checks the two properties everything rests on: the
model only ever sees retrieved chunks, and every citation resolves to a chunk
that was actually retrieved.

## When the document cannot answer

Two routes to a refusal:

1. **Nothing retrieved** — the model is not called at all.
2. **The model says the evidence is not enough**, returning `answerable: false`.

There is no similarity threshold, on purpose: cosine scores are poorly calibrated
across models and documents, and a fixed cutoff turns into false refusals. The
model sees the evidence and is better placed to judge.

That is a decision, not a certainty, so every request logs its top retrieval
score. The first real run showed answerable questions clustering at a median of
0.68 and unanswerable ones at 0.40 — good separation, but the ranges overlap
(lowest answerable 0.47, highest unanswerable 0.52), so a threshold would trade
one hallucination for one false refusal. That is why there isn't one.

## Failure handling

| Failure | Response |
| --- | --- |
| Empty, over-long or malformed request | 422 |
| Unknown document | 404 |
| Not a PDF | 415 |
| Upload over the size limit | 413 |
| Corrupt, encrypted or text-free PDF | 422 with a specific message |
| Embedding or LLM failure | Retried once, then 502 |
| Provider timeout | 504 |
| Rejected request, such as a bad parameter | 502 at once, no retry — it fails the same way twice |
| Unparseable model output | Retried once, then a 200 refusal |
| Nothing retrieved | 200 refusal |

Failing to answer is a valid answer and returns 200. Only a broken request or a
broken dependency is an error status.

## Observability

Every `/chat` logs one JSON line: request id, rewritten query, how much was
retrieved, top score, whether it was answerable, citation count, which guards
fired, latency. `"debug": true` returns the same detail in the response,
including per-chunk scores, the evidence sent, and the raw model output.

## Evaluation

20 questions over the two PDFs in `corpus/`, scored on five metrics and broken
down by category. Retrieval is counted against gold pages; the answer is scored
by an LLM judge on four separate axes, never blended into one number. Every run
is stored with the settings that produced it, so two runs can be compared.

Method and limits: **[EVAL.md](EVAL.md)**. Latest numbers:
**[eval/results.md](eval/results.md)**.

## What the first real run showed

| Metric | Score |
| --- | --- |
| Retrieval | 84% |
| Correctness | 85% |
| Groundedness | 88% |
| Citation support | 88% |
| Abstention | 75% |

Multi-passage questions are the weak spot: 75% retrieval, 62% correctness, 50%
citation support. The system finds one of the two passages an answer needs and
answers from it. That is the first thing hybrid search or a larger top-k should
be measured against.

One hallucination: asked for the period of a circular orbit while scoped to the
calculus notes, it answered instead of refusing.

## Limitations, and what comes next

- **Maths extracts badly.** Integral signs arrive as `(cid:90)`, primes as
  `(cid:48)`. Answers quoting formulas are unreliable. A maths-aware extractor
  would be needed.
- **Tables lose their structure** — a row is read as one run-together line.
- **No OCR**, so scanned documents are rejected rather than mishandled.
- **Page numbers are PDF positions**, not the numbers printed on the page. These
  differ in any document with front matter.
- **Parsing is pure Python** and slow: minutes for a long book. It is a one-off
  cost, but it is not fast.
- **Single writer.** SQLite in WAL mode with one connection per thread is fine
  for one process; two servers sharing a data directory would contend.
- **No authentication.** Conversations are not scoped to a user, because there
  are no users.

In order: hybrid search for the multi-passage gap, a reranker for documents with
many near-identical sections, and streaming responses.

## Decisions at a glance

| Decision | Choice | Why | Cost |
| --- | --- | --- | --- |
| Retrieval | Dense only | Handles paraphrase; one moving part | Misses exact identifiers |
| Chunk size | 500 tokens, 75 overlap | Coherent passages, precise retrieval | Boundaries can still split an answer |
| Headings | Table of contents first | Authoritative, no guessing | Falls back to page-only citations |
| Embedded text | Prefixed with section title | Anchors context-free chunks | Slightly more to embed |
| Storage | SQLite, vectors in the row | One transaction, deletable, no service | Single writer, must fit memory |
| Citations | Evidence positions, resolved by us | Invented pages cannot reach the user | Needs guard logic |
| Abstention | Model's judgement, no threshold | Cosine scores are poorly calibrated | Measured, not assumed |
| Follow-ups | Rewrite only when there is history | They fail without it | One extra model call |
| Judge | Four separate scores | Shows which part regressed | One call per question |
| Testing | Offline fakes | Fast, free, deterministic | Does not exercise the real provider |

## Two things worth recording

**The PDF library changed.** The plan called for PyMuPDF; the Windows Application
Control policy on this machine blocks its native library. Parsing uses
**pdfplumber** (text and layout) and **pypdf** (table of contents) instead, both
pure Python. Only the parsing section of `ingestion.py` changed — everything
below it works on a list of `Line` objects and was untouched, which is the
separation doing its job. The cost is speed.

**Real PDFs found bugs that generated fixtures could not.** Running the parser
over an actual document showed the column split firing on a styled title page and
cutting the centred title in half; heading rules calling 38 of 129 lines
headings, mostly wrapped sentences; and one-line "sections" becoming
28-character chunks. All three are fixed, each with a regression test.
