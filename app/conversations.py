"""Stored conversations.

The chat endpoint stays stateless: pass `history` and nothing is written. Pass a
`conversation_id` instead and the server loads the history and records both new
turns. The stateless path remains the primitive, which matters because the
evaluation uses it with canned history and must not depend on stored state.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.db import Database, loads
from app.errors import ConversationNotFound
from app.schemas import Citation, Conversation, Message, StoredMessage, Trace

log = logging.getLogger(__name__)

UNTITLED = "New chat"
TITLE_CHARS = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(db: Database, title: str = UNTITLED, doc_id: str | None = None) -> Conversation:
    conversation_id = uuid.uuid4().hex[:16]
    now = _now()
    with db.write() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, doc_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, title[:TITLE_CHARS] or UNTITLED, doc_id, now, now),
        )
    return Conversation(
        id=conversation_id, title=title[:TITLE_CHARS] or UNTITLED, doc_id=doc_id,
        created_at=now, updated_at=now, messages=[],
    )


def list_all(db: Database) -> list[Conversation]:
    rows = db.query(
        "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) "
        "AS message_count FROM conversations c ORDER BY c.updated_at DESC"
    )
    return [_conversation(row, []) for row in rows]


def get(db: Database, conversation_id: str) -> Conversation:
    row = db.one("SELECT * FROM conversations WHERE id = ?", conversation_id)
    if row is None:
        raise ConversationNotFound("No conversation with id " + conversation_id)
    messages = db.query(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", conversation_id
    )
    return _conversation(row, [_message(m) for m in messages])


def delete(db: Database, conversation_id: str) -> None:
    get(db, conversation_id)  # raises if unknown
    with db.write() as connection:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def history(db: Database, conversation_id: str, max_turns: int) -> list[Message]:
    """Prior turns, newest last, trimmed the way the stateless path trims."""
    rows = db.query(
        "SELECT role, content FROM messages WHERE conversation_id = ? "
        "ORDER BY id DESC LIMIT ?",
        conversation_id,
        max_turns * 2,
    )
    return [Message(role=row["role"], content=row["content"]) for row in reversed(rows)]


def append(
    db: Database,
    conversation_id: str,
    role: str,
    content: str,
    answerable: bool | None = None,
    citations: list[Citation] | None = None,
    trace: Trace | None = None,
) -> None:
    now = _now()
    with db.write() as connection:
        connection.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, answerable, citations, trace, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id, role, content,
                None if answerable is None else int(answerable),
                json.dumps([c.model_dump() for c in citations]) if citations else None,
                trace.model_dump_json() if trace else None,
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
        # The first question names the conversation, so the list is readable.
        if role == "user":
            connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ?",
                (content[:TITLE_CHARS], conversation_id, UNTITLED),
            )


def _conversation(row, messages: list[StoredMessage]) -> Conversation:
    keys = row.keys()
    return Conversation(
        id=row["id"],
        title=row["title"],
        doc_id=row["doc_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"] if "message_count" in keys else len(messages),
        messages=messages,
    )


def _message(row) -> StoredMessage:
    return StoredMessage(
        role=row["role"],
        content=row["content"],
        answerable=None if row["answerable"] is None else bool(row["answerable"]),
        citations=[Citation.model_validate(c) for c in loads(row["citations"], [])],
        trace=Trace.model_validate(loads(row["trace"], None)) if row["trace"] else None,
        created_at=row["created_at"],
    )
