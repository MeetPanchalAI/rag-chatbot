"""Scoring the evaluation set.

Questions run through `answer_question`, the same path a real request takes, so
the numbers describe the shipped system rather than a parallel harness.

Two kinds of measurement, deliberately kept apart:

  Retrieval is scored arithmetically against gold pages. We know which pages
  hold the answer, so a model opinion about retrieval would be strictly worse
  than counting.

  The answer is scored by an LLM judge, on four axes kept separate. One blended
  number would hide the thing worth seeing between two runs: that retrieval
  improved while citations got worse.

Each question carries the document it belongs to. That matters most for the
unanswerable questions: several are answerable from the *other* document, so
scoping each question to its own is what makes them a real abstention test.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.errors import IngestError
from app.judge import EMPTY, SCORES, JudgeResult, judge_answer
from app.pipeline import answer_question
from app.providers import Embedder, LLM
from app.schemas import ChatRequest, ChatResponse
from app.vector_store import VectorStore

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"

CATEGORIES = ("factual", "multi_passage", "follow_up", "unanswerable", "similar_sections")

Progress = Callable[[int, int], None]


def load_questions(path: Path | None = None) -> list[dict]:
    with (path or QUESTIONS_FILE).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_documents(questions: list[dict], store: VectorStore) -> dict[str, str]:
    """Map the `doc` name in the set to an indexed document id, by filename stem."""
    by_stem = {d.filename.rsplit(".", 1)[0].lower(): d.doc_id for d in store.list_documents()}
    wanted = {q["doc"] for q in questions if q.get("doc")}
    missing = sorted(name for name in wanted if name.lower() not in by_stem)
    if missing:
        raise IngestError(
            "The evaluation set names documents that are not indexed: {}. "
            "Ingest them first.".format(", ".join(missing))
        )
    return {name: by_stem[name.lower()] for name in wanted}


def covered_pages(response: ChatResponse) -> set[int]:
    pages: set[int] = set()
    for item in response.trace.retrieved if response.trace else []:
        pages.update(range(item.page_start, item.page_end + 1))
    return pages


def cited_pages(response: ChatResponse) -> set[int]:
    pages: set[int] = set()
    for citation in response.citations:
        pages.update(range(citation.page_start, citation.page_end + 1))
    return pages


def score(question: dict, response: ChatResponse, verdict: JudgeResult) -> dict:
    """Score one question.

    `recall` is the share of gold pages retrieved, not a yes/no. A question
    needing two passages that found one scores 0.5, which is what it deserves;
    a hit/miss flag would call it a success.
    """
    gold = set(question.get("gold_pages") or [])
    retrieved = covered_pages(response)
    cited = cited_pages(response)
    should_answer = bool(question.get("answerable", True))

    row = {
        "id": question["id"],
        "doc": question.get("doc", ""),
        "type": question.get("type", ""),
        "question": question["question"],
        "should_be_answerable": should_answer,
        "answerable": response.answerable,
        "answer": response.answer,
        "expected_answer": question.get("expected_answer", ""),
        "citations": [c.display for c in response.citations],
        "top_score": response.trace.top_score if response.trace else None,
        "guards": response.trace.guards if response.trace else [],
        "recall": (len(gold & retrieved) / len(gold)) if gold else None,
        "cited_gold": (len(cited & gold) / len(cited)) if cited and gold else None,
        # An unanswerable question answered anyway is the failure that matters.
        "unsupported": (not should_answer) and response.answerable,
        "judge_reason": verdict.reason,
    }
    row.update(verdict.scores)
    return row


def run_evaluation(
    questions: list[dict],
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    rewrite_llm: LLM,
    settings: Settings,
    doc_id: str | None = None,
    judge_llm: LLM | None = None,
    on_progress: Progress | None = None,
) -> list[dict]:
    """Run every question. `doc_id` overrides the per-question document."""
    documents = resolve_documents(questions, store) if doc_id is None else {}
    rows: list[dict] = []

    for index, question in enumerate(questions, start=1):
        scope = doc_id or documents.get(question.get("doc", ""))
        request = ChatRequest(
            question=question["question"],
            doc_id=scope,
            history=question.get("history") or [],
            debug=True,
        )
        response = answer_question(request, store, embedder, llm, rewrite_llm, settings)

        verdict = EMPTY
        if judge_llm is not None:
            verdict = judge_answer(
                judge_llm,
                question["question"],
                question.get("history") or [],
                question.get("expected_answer", ""),
                response.answer,
                (response.trace.evidence if response.trace else "") or "",
                [c.display for c in response.citations],
                question.get("expected_points") or [],
            )

        rows.append(score(question, response, verdict))
        if on_progress:
            on_progress(index, len(questions))
    return rows


# --- metrics -----------------------------------------------------------------


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _half(value: float | None) -> float | None:
    return None if value is None else value / 2.0


def metrics(rows: list[dict]) -> dict[str, float | None]:
    """The five numbers, each 0 to 1 and each higher-is-better.

    The judge scores 0, 1 or 2; they are halved so every metric shares a scale.
    """
    answerable = [r for r in rows if r["should_be_answerable"]]
    unanswerable = [r for r in rows if not r["should_be_answerable"]]
    unsupported = _mean([float(r["unsupported"]) for r in unanswerable])

    return {
        "retrieval": _mean([r["recall"] for r in answerable]),
        "correctness": _half(_mean([r.get("correctness") for r in rows])),
        "groundedness": _half(_mean([r.get("groundedness") for r in rows])),
        "citation_support": _half(_mean([r.get("citation_support") for r in answerable])),
        "abstention": None if unsupported is None else 1.0 - unsupported,
    }


def breakdown(rows: list[dict]) -> list[dict]:
    """The same five metrics per question category."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)

    order = [c for c in CATEGORIES if c in grouped]
    order += [c for c in grouped if c not in CATEGORIES]
    return [
        {"type": name, "count": len(grouped[name]), **metrics(grouped[name])}
        for name in order
    ]


LABELS = {
    "retrieval": ("Retrieval", "share of gold pages retrieved"),
    "correctness": ("Correctness", "the answer agrees with the reference"),
    "groundedness": ("Groundedness", "every claim is supported by the evidence"),
    "citation_support": ("Citation support", "the cited page holds the claim"),
    "abstention": ("Abstention", "refused when the document could not answer"),
}


def headline(rows: list[dict]) -> list[dict]:
    """The five metrics as tiles, all phrased so higher is better."""
    values = metrics(rows)
    unanswerable = [r for r in rows if not r["should_be_answerable"]]
    unsupported = sum(1 for r in unanswerable if r["unsupported"])
    judged = sum(1 for r in rows if r.get("correctness") is not None)

    details = {
        "retrieval": "over {} answerable questions".format(
            sum(1 for r in rows if r["should_be_answerable"])
        ),
        "correctness": "judged on {} of {} questions".format(judged, len(rows)),
        "groundedness": "right for the wrong reason still scores low",
        "citation_support": "judged on answerable questions only",
        "abstention": "{} of {} unanswerable questions were answered anyway".format(
            unsupported, len(unanswerable)
        ),
    }
    return [
        {
            "label": LABELS[key][0],
            "metric": LABELS[key][1],
            "value": values[key],
            "detail": details[key],
        }
        for key in LABELS
    ]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else "{:.0%}".format(value)


def summarise(rows: list[dict]) -> dict[str, str]:
    values = metrics(rows)
    unanswerable = [r for r in rows if not r["should_be_answerable"]]
    unsupported = sum(1 for r in unanswerable if r["unsupported"])
    judged = sum(1 for r in rows if r.get("correctness") is not None)

    summary = {"Questions": str(len(rows))}
    for key, (label, _) in LABELS.items():
        summary[label] = _pct(values[key])
    summary["Unsupported-answer rate"] = (
        "{} ({}/{})".format(
            _pct(unsupported / len(unanswerable)), unsupported, len(unanswerable)
        )
        if unanswerable
        else "n/a"
    )
    summary["Answers judged"] = "{}/{}".format(judged, len(rows))
    return summary


def score_ranges(rows: list[dict]) -> list[dict]:
    """Top retrieval score per group, for the separation plot."""
    out: list[dict] = []
    for label, wanted in (("answerable", True), ("unanswerable", False)):
        scores = sorted(
            r["top_score"]
            for r in rows
            if r["should_be_answerable"] is wanted and r["top_score"] is not None
        )
        if scores:
            out.append({
                "group": label, "min": min(scores), "median": scores[len(scores) // 2],
                "max": max(scores), "count": len(scores),
            })
    return out


def score_distribution(rows: list[dict]) -> list[str]:
    return [
        "- {}: min {:.3f}, median {:.3f}, max {:.3f} (n={})".format(
            r["group"], r["min"], r["median"], r["max"], r["count"]
        )
        for r in score_ranges(rows)
    ]


def failures(rows: list[dict]) -> list[dict]:
    """Anything worth reading by hand: a wrong verdict, or a less than full score."""
    return [
        r
        for r in rows
        if r["should_be_answerable"] != r["answerable"]
        or (r["recall"] is not None and r["recall"] < 1.0)
        or any(r.get(name) is not None and r[name] < 2 for name in SCORES)
    ]


def write_report(rows: list[dict], path: Path) -> None:
    out = ["# Evaluation results", "", "## Summary", "", "| Metric | Result |", "| --- | --- |"]
    out += ["| {} | {} |".format(k, v) for k, v in summarise(rows).items()]

    out += ["", "## By category", "",
            "| Category | n | Retrieval | Correctness | Groundedness | Citations | Abstention |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for group in breakdown(rows):
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            group["type"].replace("_", " "), group["count"],
            _pct(group["retrieval"]), _pct(group["correctness"]),
            _pct(group["groundedness"]), _pct(group["citation_support"]),
            _pct(group["abstention"])))

    out += ["", "## Top retrieval score", ""] + (
        score_distribution(rows) or ["- no scores recorded"]
    )

    out += ["", "## Per question", "",
            "| ID | Doc | Type | Expected | Result | Recall | Corr | Grnd | Cite |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["id"], row["doc"], row["type"].replace("_", " "),
            "answer" if row["should_be_answerable"] else "refuse",
            "answered" if row["answerable"] else "refused",
            "-" if row["recall"] is None else "{:.0%}".format(row["recall"]),
            _score(row.get("correctness")), _score(row.get("groundedness")),
            _score(row.get("citation_support"))))

    bad = failures(rows)
    out += ["", "## Worth a look", ""]
    out += [
        "- **{}** ({}): {}{}".format(
            r["id"], r["type"].replace("_", " "), r["question"],
            " - " + r["judge_reason"] if r["judge_reason"] else "",
        )
        for r in bad
    ] or ["Nothing scored below full marks."]

    counts = Counter(g.split(":")[0] for r in rows for g in r["guards"])
    if counts:
        out += ["", "## Guards triggered", ""]
        out += ["- {}: {}".format(name, count) for name, count in counts.items()]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _score(value) -> str:
    return "-" if value is None else str(value)
