# AI usage

## Tools

- **Claude Code** (Claude Opus 5) — the design review, the implementation, the
  tests, and this documentation.
- **ChatGPT** — drafting the first version of the design plan.

## What they were used for

The plan was drafted with ChatGPT, then reviewed against the brief before any
code was written. Claude Code wrote the modules, the test suite, the evaluation
harness and the docs. Every design decision was reviewed; the three below are
where the AI suggestion was wrong or weaker than the alternative.

## Where an AI suggestion was wrong

### Temperature 0 on a reasoning model

The drafted plan set `temperature: 0` for the answering model, reasoning that
grounded extraction should be deterministic. Sound for an ordinary chat model,
wrong for this one: `gpt-5.6-luna` is a reasoning model and accepts only its
default temperature. Sending `0` fails the request.

It would not have shown up in the tests either, because they use a fake LLM. It
would have failed on the first real call.

Changed to `LLM_TEMPERATURE=1.0`, configurable, with `LLM_REASONING_EFFORT`
blank-able so the parameter is omitted for models that reject it.

### A metric that would have hidden the failures it was for

The plan scored retrieval as "did any retrieved chunk cover a gold page". For a
question needing two passages, retrieving one scores a full hit — so the metric
would look healthiest exactly where multi-passage retrieval was failing.

Changed to the share of gold pages retrieved, so finding one of two scores 0.5.
The first real run then showed multi-passage retrieval at 75% against 100% for
factual questions, which is the gap the old metric would have hidden.

### Identifying documents by filename

The evaluation set names the document each question belongs to, and the first
implementation matched those names to indexed documents by filename. Uploading
the same PDFs under different names broke the whole run with "the evaluation set
names documents that are not indexed", even though the indexed bytes were
identical.

Filename is a name a person chooses and can change, so it is the wrong key. A
document id is a hash of the file contents, and the corpus is in the repository,
so matching now goes by content first and falls back to filename and title.

## Where AI was clearly worth it

Running the parser over a real PDF rather than a generated fixture found three
defects at once — a column split cutting a centred title in half, heading rules
classifying 38 of 129 lines as headings, and one-line sections becoming
28-character chunks. Diagnosing and fixing all three with regression tests took
minutes rather than an afternoon.
