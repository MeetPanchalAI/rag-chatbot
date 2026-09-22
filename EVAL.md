# Evaluation

25 questions, five in each of the five behaviours the brief names. Every run
goes through the same path a real request takes, and every run is stored, so a
change to the system can be compared against the run before it.

## The set

`eval/questions.jsonl`, one JSON object per line.

| Field | Meaning |
| --- | --- |
| `id`, `type` | Identifier and category |
| `doc` | Which document the question is asked of |
| `question` | What to ask |
| `history` | Prior turns, for follow-ups |
| `answerable` | Whether that document can answer it |
| `gold_pages` | Pages holding the answer |
| `expected_answer` | The reference answer the judge grades against |

| Category | n | What it tests |
| --- | --- | --- |
| `factual` | 5 | Evidence sits in one place |
| `multi_passage` | 5 | The answer needs two passages, often two pages |
| `follow_up` | 5 | The question is meaningless without the turn before it |
| `unanswerable` | 5 | The document cannot answer; the system must say so |
| `similar_sections` | 5 | Neighbouring passages compete for the same query |

Two documents: `free221.pdf` (calculus notes, 134 pages) and `Lecture10.pdf`
(physics lecture, 26 pages).

**Every question is scoped to its own document, and that is load-bearing.** The
unanswerable questions are unanswerable *from the document they are asked of*,
and answerable from the other one. "What work does a 5 N force do?" is not in
the calculus notes but is in the physics lecture; the epsilon-delta definition
is the reverse. Run the set across both documents at once and those five stop
testing anything. Scoped, they test abstention and cross-document leakage at the
same time.

**The follow-ups carry real history.** *"Then how do the notes use it to compute
a definite integral?"* has no antecedent on its own. If query rewriting breaks,
these fail, which is the point of them.

**Gold pages were checked against the extracted text**, not against the PDF as
read on screen. Page numbers here are PDF positions, and a document with front
matter shifts them. A mislabelled gold page produces a confident, meaningless
number.

## What is measured

Five metrics, each 0–1, each higher-is-better so one bar means one thing.

| Metric | How | Answers |
| --- | --- | --- |
| **Retrieval** | Arithmetic: share of gold pages retrieved | Did retrieval find the evidence? |
| **Correctness** | Judge 0/1/2 | Is the answer right? |
| **Groundedness** | Judge 0/1/2 | Is it supported by the evidence shown? |
| **Citation support** | Judge 0/1/2 | Does the cited page hold the claim? |
| **Abstention** | Arithmetic: 1 − unsupported-answer rate | Does it refuse when it should? |

All five are also reported **per category**, which is the view that says *where*
the system is weak. A single set of averages cannot.

**Retrieval is scored arithmetically, never by the judge.** We know which pages
hold the answer, so counting beats an opinion. Recall is a share, not a hit or
miss: a question needing two passages that found one scores 0.5. A yes/no flag
would call that a success, which is exactly where multi-passage retrieval fails
quietly.

**Groundedness is separate from correctness on purpose.** A model can give the
right answer from its own knowledge while the document says nothing. That is
correct and ungrounded, and it is a failure of this system even though the
answer is true.

**Unsupported-answer rate** is reported outright as well as inverted into
Abstention, because it is the safety number: how often the system answered a
question its document could not support.

## The judge

An LLM scores the answer only, on four axes each 0 (poor), 1 (partial), 2
(good): correctness, groundedness, citation_support, completeness. It sees the
question, the history, the reference answer, the evidence that was retrieved,
and the citations.

There is deliberately **no single blended score**. One number would hide the
thing worth seeing between two runs: that retrieval improved while citations got
worse.

A judge that returns unparseable output, an out-of-range score, or fails
outright leaves the question **unscored** rather than defaulted. A default would
move the averages and look like a change in the system.

`judge: false` on a run skips it, giving retrieval and abstention only at no
extra model cost.

## Running it

```bash
python -m eval.run_eval                 # or the Evaluation tab in the UI
python -m eval.run_eval --no-judge      # retrieval metrics only
```

Both write `eval/results.md` and `eval/results.jsonl`, and store the run in the
database with the settings that produced it — model, chunk size, overlap, top-k,
context budget, reasoning effort. Two runs can then be compared and the
difference attributed rather than guessed at.

Cost is one or two model calls per question, plus one judge call.

## Known limits

- **Mathematics extracts badly from PDFs.** Integral signs arrive as
  `(cid:90)`, primes as `(cid:48)`. Questions about formulas are graded against
  evidence that is partly unreadable, and the numbers will show it. This is a
  real property of the system, not an artefact of the evaluation.
- **Page-level gold labels are coarse for slide decks.** One `Lecture10` chunk
  spans eight pages, so retrieval there is scored generously.
- **The judge is a model**, with its own bias, and it grades against a reference
  answer a human wrote. It is a consistent instrument for comparing runs, not
  ground truth.
- **25 questions is small.** A one-question change moves a category metric by 20
  percentage points. Treat category numbers as direction, not precision.
