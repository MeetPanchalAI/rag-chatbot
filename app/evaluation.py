"""Scoring the evaluation set.

Questions run through `answer_question`, the same path a real request takes, so
the numbers describe the shipped system rather than a parallel harness.

This lives in the app rather than in the eval script so the API can run it too.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.pipeline import answer_question
from app.providers import Embedder, LLM
from app.schemas import ChatRequest, ChatResponse
from app.vector_store import VectorStore

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"

Progress = Callable[[int, int], None]


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or QUESTIONS_FILE
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def covered_pages(response: ChatResponse) -> set[int]:
    """Every page the retrieved chunks span."""
    pages: set[int] = set()
    for item in response.trace.retrieved if response.trace else []:
        pages.update(range(item.page_start, item.page_end + 1))
    return pages


def cited_pages(response: ChatResponse) -> set[int]:
    pages: set[int] = set()
    for citation in response.citations:
        pages.update(range(citation.page_start, citation.page_end + 1))
    return pages


def score(question: dict, response: ChatResponse) -> dict:
    """Score one question.

    Retrieval and citations are scored against gold pages. Answer wording is
    screened by keyword and left to a human to judge properly.
    """
    gold = set(question.get("gold_pages") or [])
    retrieved = covered_pages(response)
    cited = cited_pages(response)
    expected = [k.lower() for k in question.get("expected_answer_contains") or []]
    answer = response.answer.lower()

    return {
        "id": question["id"],
        "type": question.get("type", ""),
        "question": question["question"],
        "should_be_answerable": bool(question.get("answerable", True)),
        "answerable": response.answerable,
        "answer": response.answer,
        "citations": [c.display for c in response.citations],
        "top_score": response.trace.top_score if response.trace else None,
        "guards": response.trace.guards if response.trace else [],
        # Did we find any page holding the answer, and did we find all of them?
        # The second is what multi-passage questions actually test.
        "recall": bool(gold & retrieved) if gold else None,
        "coverage": gold.issubset(retrieved) if gold else None,
        # Does the page we cited actually hold the answer?
        "citation_precision": (len(cited & gold) / len(cited)) if cited and gold else None,
        "keywords_found": all(k in answer for k in expected) if expected else None,
    }


def run_evaluation(
    questions: list[dict],
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    rewrite_llm: LLM,
    settings: Settings,
    doc_id: str | None = None,
    on_progress: Progress | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for index, question in enumerate(questions, start=1):
        request = ChatRequest(
            question=question["question"],
            doc_id=doc_id,
            history=question.get("history") or [],
            debug=True,
        )
        response = answer_question(request, store, embedder, llm, rewrite_llm, settings)
        rows.append(score(question, response))
        if on_progress:
            on_progress(index, len(questions))
    return rows


def summarise(rows: list[dict]) -> dict:
    def ratio(values: list[bool]) -> str:
        if not values:
            return "n/a"
        return "{:.0%} ({}/{})".format(sum(values) / len(values), sum(values), len(values))

    answerable = [r for r in rows if r["should_be_answerable"]]
    unanswerable = [r for r in rows if not r["should_be_answerable"]]
    precisions = [
        r["citation_precision"] for r in answerable if r["citation_precision"] is not None
    ]

    return {
        "Questions": str(len(rows)),
        "Retrieval recall (any gold page found)": ratio(
            [r["recall"] for r in answerable if r["recall"] is not None]
        ),
        "Retrieval coverage (all gold pages found)": ratio(
            [r["coverage"] for r in answerable if r["coverage"] is not None]
        ),
        "Answered when it should be": ratio([r["answerable"] for r in answerable]),
        "Refused when it should be": ratio([not r["answerable"] for r in unanswerable]),
        "Hallucination rate (answered the unanswerable)": ratio(
            [r["answerable"] for r in unanswerable]
        ),
        "Answers with a citation": ratio(
            [bool(r["citations"]) for r in answerable if r["answerable"]]
        ),
        "Citation precision (cited page holds the answer)": (
            "{:.0%}".format(sum(precisions) / len(precisions)) if precisions else "n/a"
        ),
        "Expected keywords present": ratio(
            [r["keywords_found"] for r in answerable if r["keywords_found"] is not None]
        ),
    }


def _share(values: list[bool]) -> tuple[float | None, int, int]:
    if not values:
        return None, 0, 0
    return sum(values) / len(values), sum(values), len(values)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else "{:.0%}".format(value)


def headline(rows: list[dict]) -> list[dict]:
    """The four things the brief asks you to measure, as numbers.

    `summarise` formats for the written report; this is for the UI, which needs
    the value to draw a meter. All four are phrased so that higher is better, so
    the meters mean the same thing in every tile. The hallucination count is the
    detail under Refusal rather than a tile of its own: a filling bar for a
    number you want at zero reads backwards.
    """
    answerable = [r for r in rows if r["should_be_answerable"]]
    unanswerable = [r for r in rows if not r["should_be_answerable"]]

    recall, recall_hit, recall_of = _share(
        [r["recall"] for r in answerable if r["recall"] is not None]
    )
    coverage, _, _ = _share([r["coverage"] for r in answerable if r["coverage"] is not None])
    keywords, keyword_hit, keyword_of = _share(
        [r["keywords_found"] for r in answerable if r["keywords_found"] is not None]
    )
    precisions = [
        r["citation_precision"] for r in answerable if r["citation_precision"] is not None
    ]
    citation = sum(precisions) / len(precisions) if precisions else None
    refusal, refused, refusal_of = _share([not r["answerable"] for r in unanswerable])
    hallucinated = sum(1 for r in unanswerable if r["answerable"])

    return [
        {
            "label": "Retrieval",
            "metric": "a page holding the answer was retrieved",
            "value": recall,
            "detail": "{}/{} questions - every gold page found for {}".format(
                recall_hit, recall_of, _pct(coverage)
            ),
        },
        {
            "label": "Answer",
            "metric": "the expected content is in the answer",
            "value": keywords,
            "detail": "{}/{} questions - a keyword screen, not a judgement".format(
                keyword_hit, keyword_of
            ),
        },
        {
            "label": "Citation",
            "metric": "the cited page holds the answer",
            "value": citation,
            "detail": "mean over answered questions",
        },
        {
            "label": "Refusal",
            "metric": "said so when the document could not answer",
            "value": refusal,
            "detail": "{}/{} questions - {} answered that should not have been".format(
                refused, refusal_of, hallucinated
            ),
        },
    ]


def score_distribution(rows: list[dict]) -> list[str]:
    """Top retrieval score, split by whether the document can answer at all.

    We deliberately apply no similarity threshold and let the model declare when
    the evidence is insufficient. These two distributions are the evidence for
    or against that choice: if they separate cleanly, a threshold would be a
    cheap extra safeguard.
    """
    lines = []
    for label, wanted in (("answerable", True), ("unanswerable", False)):
        scores = [
            r["top_score"]
            for r in rows
            if r["should_be_answerable"] is wanted and r["top_score"]
        ]
        if scores:
            lines.append(
                "- {}: min {:.3f}, median {:.3f}, max {:.3f} (n={})".format(
                    label,
                    min(scores),
                    sorted(scores)[len(scores) // 2],
                    max(scores),
                    len(scores),
                )
            )
    return lines


def score_ranges(rows: list[dict]) -> list[dict]:
    """The same distributions as numbers, for the UI to plot."""
    out: list[dict] = []
    for label, wanted in (("answerable", True), ("unanswerable", False)):
        scores = sorted(
            r["top_score"]
            for r in rows
            if r["should_be_answerable"] is wanted and r["top_score"] is not None
        )
        if scores:
            out.append(
                {
                    "group": label,
                    "min": min(scores),
                    "median": scores[len(scores) // 2],
                    "max": max(scores),
                    "count": len(scores),
                }
            )
    return out


def failures(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if (r["should_be_answerable"] and not r["answerable"])
        or (not r["should_be_answerable"] and r["answerable"])
        or r["recall"] is False
        or r["keywords_found"] is False
    ]


def write_report(rows: list[dict], path: Path) -> None:
    summary = summarise(rows)

    out = ["# Evaluation results", "", "## Summary", "", "| Metric | Result |", "| --- | --- |"]
    out += ["| {} | {} |".format(k, v) for k, v in summary.items()]

    out += ["", "## Top retrieval score", ""] + (
        score_distribution(rows) or ["- no scores recorded"]
    )

    out += [
        "",
        "## Per question",
        "",
        "| ID | Type | Answerable | Recall | Citations |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        out.append(
            "| {} | {} | {} | {} | {} |".format(
                row["id"],
                row["type"],
                "yes" if row["answerable"] else "no",
                {True: "hit", False: "miss", None: "n/a"}[row["recall"]],
                "; ".join(row["citations"]) or "-",
            )
        )

    out += ["", "## Failures", ""]
    out += [
        "- **{}** ({}): {}".format(r["id"], r["type"], r["question"]) for r in failures(rows)
    ] or ["None."]

    counts = Counter(g.split(":")[0] for r in rows for g in r["guards"])
    if counts:
        out += ["", "## Guards triggered", ""]
        out += ["- {}: {}".format(name, count) for name, count in counts.items()]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
