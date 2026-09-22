# AI usage

## Tools

- **Claude Code** (Claude Opus 5) — the main tool. Used for the design review,
  the implementation, and the tests.
- **ChatGPT** — used earlier to draft the first version of the design plan.

## What they were used for

- Drafting the initial design plan, then reviewing it against the brief before
  any code was written.
- Writing the modules, the test suite and the evaluation harness.
- Writing this documentation.

Every design decision was reviewed and several were changed. The three below
are the ones where the AI suggestion was wrong or weaker than the alternative.

## Where an AI suggestion was wrong

### Temperature 0 on a reasoning model

The drafted plan specified `temperature: 0` for the answering model, with the
reasoning that grounded extraction should be deterministic. The logic is sound
for an ordinary chat model and wrong for this one: `gpt-5.6-luna` is a reasoning
model and accepts only its default temperature. Sending `0` fails the request.

It would not have shown up in the test suite either, because the tests use a
fake LLM. It would have failed on the first real call.

Changed to `LLM_TEMPERATURE=1.0`, configurable, and `LLM_REASONING_EFFORT`
made blank-able so the parameter is omitted entirely for models that reject it.

### Heading detection by font size first

The plan's heading strategy was font-size and bold heuristics: a line is a
heading if it is more than 1.15 times the median body size. That works, but it
guesses at something most PDFs state outright — they carry their own table of
contents.

Reordered so the built-in table of contents is tried first, numbered headings
second, and font heuristics only as a fallback. Section labels in citations are
now taken from the document where the document provides them, instead of being
inferred from typography in every case.

### An evaluation metric that would have hidden the failures it was for

The plan measured retrieval as "did any retrieved chunk cover a gold page".
For a question that needs two passages, retrieving one of them scores a full
hit — so the metric would look healthiest exactly where multi-passage retrieval
was failing.

Split into two: recall (any gold page found) and coverage (all gold pages
found). Citation precision was added at the same time, since the gold page
labels were already there and made it nearly free.
