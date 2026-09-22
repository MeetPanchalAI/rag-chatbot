"""SQLite storage.

One file, `app.db`, holding documents, chunks with their vectors, conversations
and evaluation runs. The original PDFs stay on disk beside it: a PDF is only
ever read whole, and blobs in a table you query make every scan and backup
carry weight for nothing.

Vectors, by contrast, belong in the chunk row. They used to live in a separate
numpy file aligned to a JSONL file by line order, with nothing enforcing that
alignment, which is why deleting a document was impossible. In one row they are
written in the same transaction as the text and deleted with it.

Connections are per thread, because the evaluation runs on a worker.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    title       TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    chunks      INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    source_file TEXT            -- the kept PDF, relative to the data directory
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    section    TEXT,
    page_start INTEGER NOT NULL,
    page_end   INTEGER NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    doc_id     TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    answerable      INTEGER,
    citations       TEXT,
    trace           TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         TEXT NOT NULL,
    question   TEXT NOT NULL,
    doc_id     TEXT,
    rewritten  TEXT,
    retrieved  INTEGER,
    evidence   INTEGER,
    top_score  REAL,
    answerable INTEGER,
    citations  INTEGER,
    guards     TEXT,
    latency_ms INTEGER
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    status      TEXT NOT NULL,          -- running | done | failed
    doc_id      TEXT,
    total       INTEGER NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    summary     TEXT,
    headline    TEXT,
    ranges      TEXT,
    knobs       TEXT,                   -- the settings that produced these numbers
    error       TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id          TEXT NOT NULL,
    type                 TEXT,
    question             TEXT,
    should_be_answerable INTEGER,
    answerable           INTEGER,
    answer               TEXT,
    citations            TEXT,
    top_score            REAL,
    recall               REAL,
    guards               TEXT
);
CREATE INDEX IF NOT EXISTS results_by_run ON eval_results(run_id);
"""


class Database:
    """A SQLite file with one connection per thread."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection = connection
        return connection

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A serialised write transaction, rolled back if the body raises."""
        connection = self.connect()
        with self._write_lock:
            try:
                with connection:
                    yield connection
            except Exception:
                log.exception("Rolling back a failed write")
                raise

    def query(self, sql: str, *params) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def one(self, sql: str, *params) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def setup(self) -> None:
        with self.write() as connection:
            connection.executescript(SCHEMA)
            for table, columns in ADDED_COLUMNS.items():
                _add_missing_columns(connection, table, columns)

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None


# Columns added after the first release. Existing databases are widened in
# place rather than rebuilt, so a run history survives an upgrade.
ADDED_COLUMNS = {
    "eval_runs": {"breakdown": "TEXT"},
    "eval_results": {
        "doc": "TEXT",
        "expected_answer": "TEXT",
        "unsupported": "INTEGER",
        "correctness": "INTEGER",
        "groundedness": "INTEGER",
        "citation_support": "INTEGER",
        "completeness": "INTEGER",
        "judge_reason": "TEXT",
    },
}


def _add_missing_columns(connection, table: str, columns: dict[str, str]) -> None:
    present = {row["name"] for row in connection.execute("PRAGMA table_info({})".format(table))}
    for name, declaration in columns.items():
        if name not in present:
            connection.execute(
                "ALTER TABLE {} ADD COLUMN {} {}".format(table, name, declaration)
            )
            log.info("Added column %s.%s", table, name)


def loads(value: str | None, fallback):
    """Read a JSON column, tolerating null and anything unparseable."""
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return fallback
