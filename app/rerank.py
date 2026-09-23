"""Reranking retrieved candidates with the LLM.

Dense and keyword search rank by similarity to the question, which is not the
same as usefulness for answering it. Reranking asks the model to read the
candidates and put the useful ones first.

One extra call per question, so it is off by default. A cross-encoder would be
cheaper per query but means a model download and a new dependency; this reuses
the provider already configured.
"""

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from app import prompts
from app.providers import LLM
from app.schemas import RetrievedChunk

log = logging.getLogger(__name__)

SNIPPET_CHARS = 400
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class Order(BaseModel):
    order: list[int]


def rerank(
    llm: LLM, query: str, candidates: list[RetrievedChunk], top_k: int
) -> tuple[list[RetrievedChunk], list[str]]:
    """Return the best top_k candidates, and any guards that fired.

    On any failure the original ranking is returned unchanged. A reranker that
    cannot be parsed must not be able to make retrieval worse than not having
    one.
    """
    if len(candidates) <= 1 or top_k <= 0:
        return candidates[:top_k], []

    listing = "\n\n".join(
        "[{}] {}".format(i, item.chunk.text[:SNIPPET_CHARS].replace("\n", " "))
        for i, item in enumerate(candidates, start=1)
    )
    prompt = "Question:\n{}\n\nPassages:\n{}".format(query, listing)

    try:
        raw = llm.complete(prompts.load("rerank"), prompt, json_mode=True)
        parsed = Order.model_validate_json(_FENCE.sub("", raw.strip()).strip())
    except Exception as exc:
        log.warning("rerank failed, keeping the original order: %s", exc)
        return candidates[:top_k], ["rerank_failed"]

    guards: list[str] = []
    chosen: list[RetrievedChunk] = []
    seen: set[int] = set()
    for position in parsed.order:
        if 1 <= position <= len(candidates) and position not in seen:
            seen.add(position)
            chosen.append(candidates[position - 1])
        else:
            guards.append("rerank_dropped:{}".format(position))

    # The model may return fewer than we need. Top up from the original order
    # rather than sending the model less evidence than it could have had.
    if len(chosen) < top_k:
        for position, item in enumerate(candidates, start=1):
            if position not in seen and len(chosen) < top_k:
                chosen.append(item)
    return chosen[:top_k], guards
