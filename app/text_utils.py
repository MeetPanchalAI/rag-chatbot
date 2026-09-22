"""Small text helpers shared by ingestion and retrieval."""

import re

_WS = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English prose.

    Chunk sizes and context budgets only need to be approximately right, and
    this avoids a tokenizer dependency that would download files at first use
    and break offline tests.
    """
    return max(1, len(text) // 4)


def normalize(text: str) -> str:
    """Collapse whitespace and lowercase, for comparing headings to TOC titles."""
    return _WS.sub(" ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]
