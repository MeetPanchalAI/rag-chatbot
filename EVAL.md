# Evaluation

20 questions, four in each of the five behaviours the brief names, over the two
PDFs in [`corpus/`](corpus/). Every question runs through the same path a real
request takes, and every run is stored, so a change to the system can be
compared against the run before it.

## Corpus

| File | Pages | Subject |
| --- | --- | --- |
| `free221.pdf` | 134 | First-semester calculus notes |
| `Lecture10.pdf` | 26 | Physics lecture: circular motion, work, energy |

Both are committed so the evaluation can be reproduced and read against its
source.

## The set

`eval/questions.jsonl`, one JSON object per line.

| Field | Meaning |
| --- | --- |
| `id`, `type` | Identifier and category |
| `doc` | Which document the question is asked of |
| `question`, `history` | The question, and prior turns for follow-ups |
| `answerable` | Whether that document can answer it |
| `gold_pages` | Pages holding the answer |
| `expected_answer` | Reference answer the judge grades against |
| `expected_points` | The specific facts a full answer must contain |

| Category | n | What it tests |
| --- | --- | --- |
| `factual` | 4 | Evidence sits in one place |
| `multi_passage` | 4 | The answer needs two passages |
| `follow_up` | 4 | The question depends on the turn before it |
| `unanswerable` | 4 | The document cannot answer; the system must say so |
| `similar_sections` | 4 | Neighbouring passages compete for the same query |

**Each question is scoped to its own document, and that is load-bearing.** The
four unanswerable questions are answerable from the *other* document: the work
formula is in the physics lecture but not the calculus notes, and the ε–δ
definition is the reverse. Run the set across both documents at once and those
four stop testing anything. Scoped, they test abstention and cross-document
leakage together.

**Gold pages were checked against the extracted text**, not the PDF as read on
screen — page numbers here are PDF positions. That check moved q02 from pages
48–49 to 49–50: page 48 is the product rule, and the power rule is derived on 49
and stated on 50.

## What is measured

Five metrics, each 0–1, each higher-is-better so one bar means one thing.
Reported overall and per category.

| Metric | How | Answers |
| --- | --- | --- |
| **Retrieval** | Share of gold pages retrieved | Did retrieval find the evidence? |
| **Correctness** | Judge 0/1/2 | Is the answer right? |
| **Groundedness** | Judge 0/1/2 | Is it supported by the evidence shown? |
| **Citation support** | Judge 0/1/2 | Does the cited page hold the claim? |
| **Abstention** | 1 − unsupported-answer rate | Does it refuse when it should? |

**Retrieval is arithmetic, never judged.** We know which pages hold the answer,
so counting beats an opinion. It is a share, not a hit or miss: a question
needing two passages that found one scores 0.5. A yes/no flag would call that a
success, which is where multi-passage retrieval fails quietly.

**Groundedness is separate from correctness.** A model can be right from its own
knowledge while the document says nothing. That is a failure of this system even
though the answer is true.

**Unsupported-answer rate** is the safety number, reported outright as well as
inverted into Abstention.

## The judge

An LLM scores the answer only, on four axes each 0/1/2: correctness,
groundedness, citation_support, completeness. It sees the question, the history,
the reference answer, the expected points, the retrieved evidence, and the
citations.

No single blended score, on purpose — one number would hide the thing worth
seeing between two runs: that retrieval improved while citations got worse.

A judge that returns unparseable output, an out-of-range score, or fails
outright leaves the question **unscored** rather than defaulted. A default would
move the averages and look like a change in the system.

## Running it

```bash
python -m app.ingest_cli corpus/free221.pdf corpus/Lecture10.pdf
python -m eval.run_eval                 # or the Evaluation tab
python -m eval.run_eval --no-judge      # retrieval metrics only, no judge calls
```

Each run is stored with the settings that produced it — models, chunk size,
overlap, top-k, context budget, reasoning effort, and whether hybrid search and
reranking were on — so two runs can be compared and the difference attributed
rather than guessed at.

To measure hybrid search or reranking, run once with each off, then again with
one on, and label the runs. One caveat: with hybrid search on, the retrieval
score is a fusion score rather than a cosine similarity, so `top_score` is
comparable within a retrieval mode but not across modes. Every other metric is.

Reports are written to `eval/results.md` and `eval/results.jsonl`.

Cost is one or two model calls per question, plus one judge call.

## Known limits

- **Maths extracts badly.** Integral signs arrive as `(cid:90)`, primes as
  `(cid:48)`. Questions about formulas are graded against partly unreadable
  evidence, and the numbers will show it. That is a real property of the
  system, not an artefact of the evaluation.
- **The follow-up questions read as self-contained.** They test that history
  does not corrupt an already-clear question, rather than resolution of a truly
  ambiguous one. Shortening them to "How would I apply it to sin(2x)?" would
  make the category a sharper test of query rewriting.
- **Page-level gold labels are coarse for the lecture deck.** One `Lecture10`
  chunk spans eight pages, so retrieval there is scored generously.
- **The judge is a model**, with its own bias, grading against a human-written
  reference. It is a consistent instrument for comparing runs, not ground truth.
- **20 questions is small.** One question moves a category metric by 25 points.
  Read category numbers as direction, not precision.
