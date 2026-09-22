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
| `rerank.py` | Reorders candidates before answering |
| `judge.py` | Scores answers during evaluation |
| `prompts.py` | Reads the prompt files |
| `activity.py` | The record behind the dashboard |
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

Dense search handles paraphrase, which is how people ask questions. Its weakness
is exact identifiers — codes, names, amounts — that embeddings blur together.

**Hybrid search** adds BM25 over the same chunks and fuses the two with
reciprocal rank fusion: each ranker contributes `1/(60 + rank)`. Fusing on *rank*
rather than score is the point — a cosine similarity and a BM25 score are not on
the same scale and cannot be added, but their orderings can be combined. It is on
by default because it costs no extra API call, only local computation.

Two details found by running it on the real corpus. A chunk counts as a keyword
match if it *contains a query term*, not if its score is positive: when a term
appears in most of a small corpus, BM25 gives it a negative weight, and filtering
on score would drop real matches. And single-character query terms are ignored —
in a maths text `k` and `n` appear on nearly every page, so a query containing
one matches everything and ranks noise.

**Reranking** is off by default because it costs a model call per question. When
on, retrieval fetches 20 candidates and the model orders them by usefulness
before the top 6 become evidence. Similarity to the question is not the same as
being useful for answering it. If the reranker fails or returns nonsense, the
original order is kept: a reranker must not be able to make retrieval worse than
not having one.

Both are recorded with every evaluation run, so the gain from each is measured
rather than assumed.

**Follow-ups** are rewritten into standalone queries before the search. *"What
about international applicants?"* finds nothing on its own. The rewrite costs one
small model call and only runs when there is history. A rewrite that comes back
empty or rambling is thrown away in favour of the original question. History is
used to read the question, never as evidence.

## Storage

One SQLite file, `data/app.db`: documents, chunks with their vectors,
conversations, answered questions, and evaluation runs. The original PDFs sit in
`data/documents/`.

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

## Prompts

The five instructions this system sends — answering, the retry after unparseable
output, follow-up rewriting, reranking, and judging — live in `prompts/`, one
plain text file each.

They are content, not code. They get rewritten far more often than the functions
around them, and a diff on a text file says what changed without the noise of
Python quoting. Files are read when used rather than at import, so editing one
takes effect on the next question without a restart.

Substitution is a literal `{name}` replace and never `str.format`, because
several of these prompts contain JSON examples with braces in them.

What stays in code is the user message: the evidence, the conversation and the
candidate passages, assembled under fixed labels. That is runtime data, not
instruction.

Three contracts survive any edit, and are tested: the answer prompt must ask for
*evidence numbers* rather than page numbers, which is what stops an invented page
reaching the user; it must keep the `answerable` flag, which is how refusal
works; and the four prompts sent in JSON mode must contain the word "json",
which the API requires.

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

Every `/chat` logs one JSON line — request id, rewritten query, how much was
retrieved, top score, whether it was answerable, citation count, which guards
fired, latency — and stores the same record in the database.

A log is fine for tailing and useless for answering "how often does it refuse?".
The **Activity tab** counts the last 500 questions: how many were answered, the
median latency, the median retrieval score, and which guards fired, over a list
of recent questions. Recording never blocks an answer; if the write fails the
answer still returns.

An evaluation run passes no recorder, so a benchmark does not fill the dashboard
with questions nobody asked.

`"debug": true` returns the full detail in the response: per-chunk scores, the
evidence sent, and the raw model output.

## Evaluation

20 questions over the two PDFs in `corpus/`, scored on five metrics and broken
down by category. Retrieval is counted against gold pages; the answer is scored
by an LLM judge on four separate axes, never blended into one number. Every run
is stored with the settings that produced it, so two runs can be compared.

Method and limits: **[EVAL.md](EVAL.md)**. Latest numbers:
**[eval/results.md](eval/results.md)**.

## What the runs show

Dense-only against dense + BM25, same corpus, same questions, everything else
unchanged:

| Metric | Dense | Hybrid |
| --- | --- | --- |
| Retrieval | 84% | 84% |
| Correctness | 85% | 90% |
| Groundedness | 88% | 92% |
| Citation support | 88% | 84% |
| Abstention | 75% | 100% |

**Hybrid removed the one hallucination.** Dense-only answered q14, a
circular-orbit question, from the calculus notes. With keyword search fused in,
that question retrieves nothing with matching terms, the evidence gets weaker,
and the model refuses — which is the correct behaviour.

**It helped where the baseline was weakest.** Multi-passage went from 62% to 75%
correctness and 62% to 88% groundedness.

**And it cost something.** Factual questions dropped from 100% to 75% on
correctness and citation support. A keyword match can outrank a better semantic
one for a question that was already answered well.

Retrieval is unchanged at 84% in both, and unchanged in every category. Hybrid
did not find different *pages*; it found better passages within them, and gave
the model less to work with when there was nothing to find.

With four questions per category, one question moves a category by 25 points.
Read these as direction, not precision. Reranking has not been measured yet.

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
| Retrieval | Dense + BM25, fused by rank | Catches paraphrase and exact terms | Two indexes to keep in step |
| Reranking | LLM, off by default | Relevance is not usefulness | One model call per question |
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
