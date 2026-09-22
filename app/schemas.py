"""Request, response and storage models."""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings

_s = get_settings()


# --- Storage ---


class Chunk(BaseModel):
    """One retrievable passage, with the metadata its citation is built from."""

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    section: str | None = None
    page_start: int
    page_end: int


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    title: str
    pages: int
    chunks: int
    ingested_at: str


# --- API: ingest ---


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    title: str
    pages: int
    chunks: int
    duplicate: bool


# --- API: chat ---


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=_s.max_question_chars)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_s.max_question_chars)
    doc_id: str | None = Field(
        default=None, description="Restrict the search to one document. Omit to search all."
    )
    conversation_id: str | None = Field(
        default=None,
        description="Continue a stored conversation. The server then supplies the "
        "history and records both turns. Omit it and pass `history` to stay stateless.",
    )
    history: list[Message] = Field(default_factory=list, max_length=100)
    debug: bool = False


class Citation(BaseModel):
    document: str
    page_start: int
    page_end: int
    section: str | None = None
    display: str


class RetrievalTraceItem(BaseModel):
    chunk_id: str
    score: float
    page_start: int
    page_end: int
    section: str | None = None
    used_as_evidence: bool


class Trace(BaseModel):
    request_id: str
    rewritten_query: str | None = None
    retrieved: list[RetrievalTraceItem] = Field(default_factory=list)
    top_score: float | None = None
    evidence: str | None = None
    raw_model_output: str | None = None
    guards: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation] = Field(default_factory=list)
    trace: Trace | None = None


# --- API: health ---


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int


# --- API: conversations ---


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    answerable: bool | None = None
    citations: list["Citation"] = Field(default_factory=list)
    trace: "Trace | None" = None
    created_at: str


class Conversation(BaseModel):
    id: str
    title: str
    doc_id: str | None = None
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: list[StoredMessage] = Field(default_factory=list)


class NewConversation(BaseModel):
    title: str = Field(default="New chat", max_length=200)
    doc_id: str | None = None


# --- API: activity ---


class ActivityStats(BaseModel):
    questions: int = 0
    answered: int = 0
    refused: int = 0
    answered_share: float | None = None
    median_latency_ms: float | None = None
    median_top_score: float | None = None
    guards: dict[str, int] = Field(default_factory=dict)


class ActivityItem(BaseModel):
    at: str
    question: str
    rewritten: str | None = None
    answerable: bool
    retrieved: int = 0
    citations: int = 0
    top_score: float | None = None
    latency_ms: int = 0
    guards: list[str] = Field(default_factory=list)


class Activity(BaseModel):
    stats: ActivityStats
    recent: list[ActivityItem] = Field(default_factory=list)


# --- API: evaluation ---


class EvalRequest(BaseModel):
    doc_id: str | None = None
    label: str | None = Field(default=None, max_length=120,
                              description="A note about what changed for this run.")
    judge: bool = Field(default=True,
                        description="Score answers with the LLM judge. Off gives "
                                    "retrieval metrics only, at no extra model cost.")


class EvalRunInfo(BaseModel):
    """One row in the run history."""

    run_id: int
    label: str | None = None
    status: str
    doc_id: str | None = None
    total: int
    done: int
    started_at: str
    finished_at: str | None = None
    summary: dict[str, str] | None = None
    headline: list[dict] = Field(default_factory=list)
    knobs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class EvalStatus(BaseModel):
    run_id: int | None = None
    running: bool = False
    done: int = 0
    total: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    knobs: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, str] | None = None
    headline: list[dict] = Field(default_factory=list)
    ranges: list[dict] = Field(default_factory=list)
    breakdown: list[dict] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    error: str | None = None


# --- LLM contract ---


class AnswerJSON(BaseModel):
    """The exact shape the answering model must return."""

    answer: str
    answerable: bool
    citations: list[int] = Field(default_factory=list)
