"""Stored evaluation runs.

Every run is kept, with the settings that produced it. Comparing one run to the
one before it is the whole activity of tuning a RAG system, and a report file
that gets overwritten cannot support that.
"""

import json
import logging
from datetime import datetime, timezone

from app.config import Settings
from app.db import Database, loads
from app.errors import EvalRunNotFound
from app.schemas import EvalRunInfo, EvalStatus

log = logging.getLogger(__name__)

# The settings that change what the numbers mean. Recorded per run so a
# difference between two runs can be attributed rather than guessed at.
KNOBS = (
    "llm_model",
    "embedding_model",
    "chunk_tokens",
    "chunk_overlap_tokens",
    "retriever_top_k",
    "max_context_tokens",
    "max_history_turns",
    "llm_reasoning_effort",
)


def snapshot(settings: Settings) -> dict:
    return {name: str(getattr(settings, name)) for name in KNOBS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start(db: Database, total: int, doc_id: str | None, settings: Settings, label: str | None) -> int:
    with db.write() as connection:
        cursor = connection.execute(
            "INSERT INTO eval_runs (label, status, doc_id, total, done, started_at, knobs) "
            "VALUES (?, 'running', ?, ?, 0, ?, ?)",
            (label, doc_id, total, _now(), json.dumps(snapshot(settings))),
        )
        return int(cursor.lastrowid)


def progress(db: Database, run_id: int, done: int) -> None:
    with db.write() as connection:
        connection.execute("UPDATE eval_runs SET done = ? WHERE id = ?", (done, run_id))


def finish(db: Database, run_id: int, rows: list[dict], summary: dict, headline: list,
           ranges: list, breakdown: list) -> None:
    with db.write() as connection:
        connection.execute(
            "UPDATE eval_runs SET status = 'done', finished_at = ?, summary = ?, "
            "headline = ?, ranges = ?, breakdown = ? WHERE id = ?",
            (_now(), json.dumps(summary), json.dumps(headline), json.dumps(ranges),
             json.dumps(breakdown), run_id),
        )
        for row in rows:
            connection.execute(
                "INSERT INTO eval_results (run_id, question_id, doc, type, question, "
                "should_be_answerable, answerable, answer, expected_answer, citations, "
                "top_score, recall, cited_gold, unsupported, correctness, groundedness, "
                "citation_support, completeness, judge_reason, guards) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, row["id"], row.get("doc"), row["type"], row["question"],
                    int(row["should_be_answerable"]), int(row["answerable"]),
                    row["answer"], row.get("expected_answer"),
                    json.dumps(row["citations"]), row["top_score"],
                    row["recall"], row.get("cited_gold"), int(bool(row.get("unsupported"))),
                    row.get("correctness"), row.get("groundedness"),
                    row.get("citation_support"), row.get("completeness"),
                    row.get("judge_reason"), json.dumps(row["guards"]),
                ),
            )


def fail(db: Database, run_id: int, error: str) -> None:
    with db.write() as connection:
        connection.execute(
            "UPDATE eval_runs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
            (_now(), error, run_id),
        )


def list_runs(db: Database, limit: int = 50) -> list[EvalRunInfo]:
    rows = db.query(
        "SELECT id, label, status, doc_id, total, done, started_at, finished_at, "
        "summary, headline, knobs, error FROM eval_runs ORDER BY id DESC LIMIT ?",
        limit,
    )
    return [_info(row) for row in rows]


def get_run(db: Database, run_id: int) -> EvalStatus:
    row = db.one("SELECT * FROM eval_runs WHERE id = ?", run_id)
    if row is None:
        raise EvalRunNotFound("No evaluation run with id {}".format(run_id))
    results = db.query("SELECT * FROM eval_results WHERE run_id = ? ORDER BY id", run_id)
    return EvalStatus(
        run_id=row["id"],
        running=row["status"] == "running",
        done=row["done"],
        total=row["total"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        knobs=loads(row["knobs"], {}),
        summary=loads(row["summary"], None),
        headline=loads(row["headline"], []),
        ranges=loads(row["ranges"], []),
        breakdown=loads(row["breakdown"], []),
        rows=[_result(r) for r in results],
        error=row["error"],
    )


def latest(db: Database) -> EvalStatus | None:
    row = db.one("SELECT id FROM eval_runs ORDER BY id DESC LIMIT 1")
    return get_run(db, row["id"]) if row else None


def delete(db: Database, run_id: int) -> None:
    get_run(db, run_id)  # raises if unknown
    with db.write() as connection:
        connection.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))


def _tri(value) -> int | None:
    return None if value is None else int(value)


def _bool(value) -> bool | None:
    return None if value is None else bool(value)


def _info(row) -> EvalRunInfo:
    return EvalRunInfo(
        run_id=row["id"],
        label=row["label"],
        status=row["status"],
        doc_id=row["doc_id"],
        total=row["total"],
        done=row["done"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        summary=loads(row["summary"], None),
        headline=loads(row["headline"], []),
        knobs=loads(row["knobs"], {}),
        error=row["error"],
    )


def _result(row) -> dict:
    return {
        "id": row["question_id"],
        "doc": row["doc"] or "",
        "type": row["type"],
        "question": row["question"],
        "should_be_answerable": bool(row["should_be_answerable"]),
        "answerable": bool(row["answerable"]),
        "answer": row["answer"],
        "expected_answer": row["expected_answer"] or "",
        "citations": loads(row["citations"], []),
        "top_score": row["top_score"],
        "recall": row["recall"],
        "cited_gold": row["cited_gold"],
        "unsupported": bool(row["unsupported"]),
        "correctness": row["correctness"],
        "groundedness": row["groundedness"],
        "citation_support": row["citation_support"],
        "completeness": row["completeness"],
        "judge_reason": row["judge_reason"] or "",
        "guards": loads(row["guards"], []),
    }
