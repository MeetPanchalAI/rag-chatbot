"""Answer generation, with guards between the model and the response.

Three things are enforced here, in order:
  1. the model returns parseable JSON in the agreed shape,
  2. every citation points at evidence that was actually retrieved,
  3. an answer with no surviving citation is not presented as grounded.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app import prompts
from app.providers import LLM
from app.schemas import AnswerJSON, Message

log = logging.getLogger(__name__)

REFUSAL = "The document does not contain enough information to answer that."

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class GenerationResult:
    answer: str
    answerable: bool
    citations: list[int]  # 1-based positions in the evidence list
    raw_output: str = ""
    guards: list[str] = field(default_factory=list)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def _parse(raw: str) -> AnswerJSON:
    return AnswerJSON.model_validate_json(_strip_fences(raw))


def _build_prompt(question: str, history: list[Message], evidence: str, max_turns: int) -> str:
    parts = ["Evidence:\n" + evidence]
    recent = history[-(max_turns * 2) :]
    if recent:
        transcript = "\n".join("{}: {}".format(m.role, m.content) for m in recent)
        parts.append("Conversation so far (context only, not evidence):\n" + transcript)
    parts.append("Question: " + question)
    parts.append("Reply with JSON only.")
    return "\n\n".join(parts)


def generate_answer(
    llm: LLM,
    question: str,
    history: list[Message],
    evidence: str,
    evidence_count: int,
    max_turns: int,
) -> GenerationResult:
    """Ask the model for a grounded answer and validate what comes back."""
    if evidence_count == 0:
        # Nothing was retrieved, so there is nothing to ground an answer in.
        # Calling the model here could only produce an ungrounded answer.
        return GenerationResult(REFUSAL, False, [], guards=["no_evidence"])

    prompt = _build_prompt(question, history, evidence, max_turns)
    guards: list[str] = []

    raw = llm.complete(prompts.load("answer"), prompt, json_mode=True)
    try:
        parsed = _parse(raw)
    except (ValidationError, json.JSONDecodeError, ValueError) as first_error:
        guards.append("json_retry")
        log.warning("Model returned unparseable output; retrying once: %s", first_error)
        retry_prompt = prompt + "\n\n" + prompts.load(
            "answer_retry", error=str(first_error)[:200]
        )
        raw = llm.complete(prompts.load("answer"), retry_prompt, json_mode=True)
        try:
            parsed = _parse(raw)
        except (ValidationError, json.JSONDecodeError, ValueError) as second_error:
            guards.append("json_failed")
            log.error("Model output still unparseable after retry: %s", second_error)
            return GenerationResult(REFUSAL, False, [], raw_output=raw, guards=guards)

    valid: list[int] = []
    for index in parsed.citations:
        if 1 <= index <= evidence_count and index not in valid:
            valid.append(index)
        else:
            guards.append("dropped_citation:{}".format(index))

    answerable = parsed.answerable
    answer = parsed.answer.strip()

    if answerable and not valid:
        # The model claims the evidence supports this but will not point at any
        # of it. Treat that as unsupported rather than showing it as grounded.
        guards.append("answerable_without_citations")
        return GenerationResult(REFUSAL, False, [], raw_output=raw, guards=guards)

    if not answerable:
        return GenerationResult(answer or REFUSAL, False, [], raw_output=raw, guards=guards)

    return GenerationResult(answer, True, valid, raw_output=raw, guards=guards)
