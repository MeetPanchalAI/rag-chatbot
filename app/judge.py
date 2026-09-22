"""LLM judge for generated answers.

The judge scores the answer only. Retrieval is scored deterministically against
gold pages, because we know what should have been retrieved and a model opinion
about it would be strictly worse than the arithmetic.

Four scores, each 0 (poor), 1 (partial) or 2 (good), kept separate on purpose. A
single blended number hides the thing you actually want to see between two runs:
that retrieval improved while citations got worse.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from app import prompts
from app.providers import LLM

log = logging.getLogger(__name__)

SCORES = ("correctness", "groundedness", "citation_support", "completeness")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class Verdict(BaseModel):
    correctness: int = Field(ge=0, le=2)
    groundedness: int = Field(ge=0, le=2)
    citation_support: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    reason: str = ""


@dataclass
class JudgeResult:
    scores: dict[str, int | None] = field(default_factory=dict)
    reason: str = ""
    failed: bool = False


EMPTY = JudgeResult(scores={name: None for name in SCORES})


def _prompt(question, history, expected, answer, evidence, citations, points) -> str:
    parts = []
    if history:
        parts.append(
            "Conversation before the question:\n"
            + "\n".join("{}: {}".format(m["role"], m["content"]) for m in history)
        )
    parts.append("Question:\n" + question)
    parts.append("Reference answer:\n" + expected)
    if points:
        # The specific things a full answer must contain. Grading against these
        # is steadier between runs than grading against prose alone.
        parts.append("A full answer covers:\n" + "\n".join("- " + p for p in points))
    parts.append("Evidence the system retrieved:\n" + (evidence or "(nothing was retrieved)"))
    parts.append("The system replied:\n" + answer)
    parts.append("It cited:\n" + ("; ".join(citations) if citations else "(no citation)"))
    return "\n\n".join(parts)


def judge_answer(
    llm: LLM,
    question: str,
    history: list[dict],
    expected: str,
    answer: str,
    evidence: str,
    citations: list[str],
    points: list[str] | None = None,
) -> JudgeResult:
    """Score one answer. A judge that misbehaves yields no score, never a guess."""
    prompt = _prompt(question, history, expected, answer, evidence, citations, points or [])
    try:
        raw = llm.complete(prompts.load("judge"), prompt, json_mode=True)
    except Exception as exc:
        log.warning("Judge call failed: %s", exc)
        return JudgeResult(scores=dict(EMPTY.scores), reason=str(exc)[:200], failed=True)

    try:
        verdict = Verdict.model_validate_json(_FENCE.sub("", raw.strip()).strip())
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        # An unscored question is honest. A defaulted score would quietly move
        # the averages and look like a change in the system.
        log.warning("Judge returned unusable output: %s", exc)
        return JudgeResult(scores=dict(EMPTY.scores), reason="unparseable judge output", failed=True)

    return JudgeResult(
        scores={name: getattr(verdict, name) for name in SCORES},
        reason=verdict.reason.strip(),
    )
