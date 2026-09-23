# AI usage

## Tools

- **Claude Code** (Claude Opus 5) — the design review, the implementation, the tests, and this documentation.
- **ChatGPT** — generating the EVAL SET queries and their golden answers for measuring the performance .

## What they were used for

The design plan was reviewed against the brief by claude code before any code was written. 
Claude Code wrote the modules, the test suite, the evaluation harness and the docs. 
Every design decision was reviewed; the three below are where the AI suggestion was wrong or weaker than the alternative.

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

### Over-engineering beyond the required scope

The goal was to build a focused demo within a limited time frame, not a production-ready, full-fledged product. Building a production system would have required significantly more infrastructure and engineering effort, including extensive frontend components, large-scale vector databases, Dockerized services, and additional technology choices.

A key part of the work was therefore making deliberate trade-offs—evaluating which technologies and components were actually necessary and removing unnecessary complexity—to keep the implementation aligned with the requirements and scope defined in the problem statement.

## Where AI was clearly worth it

AI significantly accelerated the implementation across multiple areas, including:

Translating the design into a working implementation
Writing comprehensive test cases
Building the UI and supporting APIs
Understanding the end-to-end workflow and execution logic
Ensuring the implementation aligned with the required standards and conventions
