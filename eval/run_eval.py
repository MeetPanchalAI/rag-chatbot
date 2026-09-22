"""Run the evaluation set and write a results report.

    python -m eval.run_eval [--questions eval/questions.jsonl] [--doc-id ID]

Every question goes through the same pipeline a real request uses, so the
numbers describe the system as shipped and not a separate test harness.

Needs a populated index and a working API key. Costs one or two model calls per
question.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from app.config import get_settings
from app.main import Providers
from app.pipeline import answer_question
from app.schemas import ChatRequest, ChatResponse
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent


def load_questions(path: Path) -> list[dict]:
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
    """Score one question. Retrieval and citations are scored against gold
    pages; answer wording is screened by keyword and reviewed by hand."""
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
        # Retrieval: did we find any gold page, and did we find all of them?
        # The second matters for questions that need several passages.
        "recall": bool(gold & retrieved) if gold else None,
        "coverage": gold.issubset(retrieved) if gold else None,
        # Citation precision: do the cited pages actually hold the answer?
        "citation_precision": (len(cited & gold) / len(cited)) if cited and gold else None,
        "keywords_found": all(k in answer for k in expected) if expected else None,
    }


def summarise(rows: list[dict]) -> dict:
    def ratio(values: list[bool]) -> str:
        if not values:
            return "n/a"
        return "{:.0%} ({}/{})".format(sum(values) / len(values), sum(values), len(values))

    answerable = [r for r in rows if r["should_be_answerable"]]
    unanswerable = [r for r in rows if not r["should_be_answerable"]]
    precisions = [r["citation_precision"] for r in answerable if r["citation_precision"] is not None]

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


def score_distribution(rows: list[dict]) -> list[str]:
    """Top retrieval score for answerable and unanswerable questions.

    We deliberately do not apply a similarity threshold, and rely on the model
    to declare when the evidence is insufficient. These two distributions are
    the evidence for or against that choice: if they separate cleanly, a
    threshold would be a cheap extra safeguard.
    """
    lines = []
    for label, wanted in (("answerable", True), ("unanswerable", False)):
        scores = [
            r["top_score"] for r in rows if r["should_be_answerable"] is wanted and r["top_score"]
        ]
        if scores:
            lines.append(
                "- {}: min {:.3f}, median {:.3f}, max {:.3f} (n={})".format(
                    label, min(scores), sorted(scores)[len(scores) // 2], max(scores), len(scores)
                )
            )
    return lines


def write_report(rows: list[dict], path: Path) -> None:
    summary = summarise(rows)
    failures = [
        r
        for r in rows
        if (r["should_be_answerable"] and not r["answerable"])
        or (not r["should_be_answerable"] and r["answerable"])
        or r["recall"] is False
        or r["keywords_found"] is False
    ]

    out = ["# Evaluation results", "", "## Summary", "", "| Metric | Result |", "| --- | --- |"]
    out += ["| {} | {} |".format(k, v) for k, v in summary.items()]

    out += ["", "## Top retrieval score", ""] + (
        score_distribution(rows) or ["- no scores recorded"]
    )

    out += ["", "## Per question", "", "| ID | Type | Answerable | Recall | Citations |", "| --- | --- | --- | --- | --- |"]
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
        "- **{}** ({}): {}".format(r["id"], r["type"], r["question"]) for r in failures
    ] or ["None."]

    counts = Counter(g.split(":")[0] for r in rows for g in r["guards"])
    if counts:
        out += ["", "## Guards triggered", ""]
        out += ["- {}: {}".format(name, count) for name, count in counts.items()]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set.")
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.jsonl")
    parser.add_argument("--doc-id", default=None, help="restrict every query to one document")
    parser.add_argument("--out", type=Path, default=ROOT / "results.md")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = VectorStore(settings.data_dir)
    store.load()
    if store.chunk_count == 0:
        print("The index is empty. Ingest a PDF first.")
        return 1

    providers = Providers(settings)
    rows: list[dict] = []

    for question in load_questions(args.questions):
        request = ChatRequest(
            question=question["question"],
            doc_id=args.doc_id,
            history=question.get("history") or [],
            debug=True,
        )
        response = answer_question(
            request, store, providers.embedder, providers.llm, providers.rewrite_llm, settings
        )
        rows.append(score(question, response))
        print("{}: {}".format(question["id"], "answered" if response.answerable else "refused"))

    write_report(rows, args.out)
    (args.out.with_suffix(".jsonl")).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    print("\nWrote {} and {}".format(args.out, args.out.with_suffix(".jsonl")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
