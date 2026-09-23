"""What the system did, question by question.

`/chat` already logged a line per request to stdout. That is fine for tailing and
useless for answering "how often does it refuse?". The same record is written to
the database so the dashboard can ask.

Statistics are computed over the most recent window rather than the whole table,
so the query stays the same size as the system gets used.
"""

import json
import logging
from datetime import datetime, timezone

from app.db import Database, loads
from app.schemas import Activity, ActivityItem, ActivityStats

log = logging.getLogger(__name__)

WINDOW = 500  # how many recent questions the statistics cover


def record(db: Database, row: dict) -> None:
    """Store one answered question. Never let this break the answer."""
    try:
        with db.write() as connection:
            connection.execute(
                "INSERT INTO queries (at, question, doc_id, rewritten, retrieved, "
                "evidence, top_score, answerable, citations, guards, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    row["question"], row["doc_id"], row["rewritten_query"],
                    row["retrieved"], row["evidence"], row["top_score"],
                    int(bool(row["answerable"])), row["citations"],
                    json.dumps(row["guards"]), row["latency_ms"],
                ),
            )
    except Exception:
        log.exception("could not record activity")


def _median(values: list[float]) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    return ordered[len(ordered) // 2] if ordered else None


def overview(db: Database, limit: int = 25) -> Activity:
    window = db.query(
        "SELECT * FROM queries ORDER BY id DESC LIMIT ?", WINDOW
    )
    answered = [r for r in window if r["answerable"]]
    guards: dict[str, int] = {}
    for row in window:
        for guard in loads(row["guards"], []):
            name = guard.split(":")[0]
            guards[name] = guards.get(name, 0) + 1

    stats = ActivityStats(
        questions=len(window),
        answered=len(answered),
        refused=len(window) - len(answered),
        answered_share=(len(answered) / len(window)) if window else None,
        median_latency_ms=_median([r["latency_ms"] for r in window]),
        median_top_score=_median([r["top_score"] for r in window]),
        guards=guards,
    )
    return Activity(stats=stats, recent=[_item(r) for r in window[:limit]])


def _item(row) -> ActivityItem:
    return ActivityItem(
        at=row["at"],
        question=row["question"],
        rewritten=row["rewritten"],
        answerable=bool(row["answerable"]),
        retrieved=row["retrieved"] or 0,
        citations=row["citations"] or 0,
        top_score=row["top_score"],
        latency_ms=row["latency_ms"] or 0,
        guards=loads(row["guards"], []),
    )
