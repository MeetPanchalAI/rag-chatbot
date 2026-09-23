"""Logging: one line per unit of work, stages behind DEBUG, ids that correlate."""

import json
import logging

from app import logs
from app.pipeline import answer_question
from app.schemas import ChatRequest
from tests.conftest import FakeEmbedder, FakeLLM


def answer(text: str, answerable: bool = True, citations: list[int] | None = None) -> str:
    return json.dumps(
        {"answer": text, "answerable": answerable, "citations": citations or [1]}
    )


def indexed(store, settings, simple_pdf) -> FakeEmbedder:
    from app.indexer import index_pdf

    embedder = FakeEmbedder()
    index_pdf(simple_pdf, "handbook.pdf", store, embedder, settings)
    return embedder


def ask(store, embedder, settings, question: str, llm: FakeLLM) -> None:
    answer_question(
        ChatRequest(question=question), store, embedder, llm, FakeLLM(), settings
    )


# --- the request id ---


def make_record() -> logging.LogRecord:
    return logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", None, None)


def test_the_request_id_is_attached_to_every_record():
    # Tested through the filter directly: configure() replaces the root handlers,
    # which would remove the one pytest captures with.
    added = logs._AddRequestId()

    with logs.request("abc123"):
        inside = make_record()
        added.filter(inside)
    outside = make_record()
    added.filter(outside)

    assert inside.request_id == "abc123"
    assert outside.request_id == "-", "a line outside any request is still formattable"


def test_the_configured_handler_carries_the_id_filter():
    logs.configure("INFO")

    handler = logging.getLogger().handlers[0]
    assert any(isinstance(f, logs._AddRequestId) for f in handler.filters)
    assert "request_id" in handler.formatter._fmt


def test_the_id_is_cleared_once_the_block_ends():
    with logs.request("abc123"):
        assert logs.current_request_id() == "abc123"

    assert logs.current_request_id() == "-"


def test_a_nested_block_restores_the_outer_id():
    with logs.request("outer"):
        with logs.request("inner"):
            assert logs.current_request_id() == "inner"
        assert logs.current_request_id() == "outer"


def test_two_questions_get_different_ids(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)
    with caplog.at_level(logging.INFO):
        ask(store, embedder, settings, "What is the maximum?", FakeLLM(answer("Fifty.")))
        ask(store, embedder, settings, "Who reviews it?", FakeLLM(answer("The office.")))

    ids = {r.request_id for r in caplog.records if r.name == "app.pipeline"}
    assert len(ids) == 2, "each question must be traceable on its own"


# --- how much is logged ---


def test_one_info_line_per_question(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)

    with caplog.at_level(logging.INFO):
        ask(store, embedder, settings, "What is the maximum?", FakeLLM(answer("Fifty.")))

    lines = [r for r in caplog.records if r.name == "app.pipeline"]
    assert len(lines) == 1, "INFO should summarise, not narrate"
    assert lines[0].levelno == logging.INFO


def test_the_summary_says_what_happened(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)

    with caplog.at_level(logging.INFO):
        ask(store, embedder, settings, "What is the maximum?", FakeLLM(answer("Fifty.")))

    message = caplog.records[-1].getMessage()
    for part in ("answered", "retrieved", "evidence", "citations", "ms"):
        assert part in message, message


def test_a_refusal_is_visible_in_the_summary(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)

    with caplog.at_level(logging.INFO):
        ask(store, embedder, settings, "Where do I park?",
            FakeLLM(answer("Not here.", answerable=False, citations=[])))

    assert "refused" in caplog.records[-1].getMessage()


def test_guards_reach_the_summary(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)

    with caplog.at_level(logging.INFO):
        ask(store, embedder, settings, "What is the maximum?",
            FakeLLM("not json", answer("Fifty.")))

    summary = [r for r in caplog.records if r.name == "app.pipeline"][-1].getMessage()
    assert "json_retry" in summary, "a guard firing must be visible without DEBUG"


def test_debug_explains_each_stage(store, settings, simple_pdf, caplog):
    embedder = indexed(store, settings, simple_pdf)

    with caplog.at_level(logging.DEBUG):
        ask(store, embedder, settings, "What is the maximum?", FakeLLM(answer("Fifty.")))

    stages = " | ".join(
        r.getMessage() for r in caplog.records if r.name == "app.pipeline"
    )
    for stage in ("asked:", "retrieved", "evidence:"):
        assert stage in stages, stages


def test_ingestion_logs_one_line_for_the_document(store, settings, simple_pdf, caplog):
    with caplog.at_level(logging.INFO):
        indexed(store, settings, simple_pdf)

    lines = [r.getMessage() for r in caplog.records if r.name == "app.indexer"]
    assert len(lines) == 1
    assert "indexed handbook.pdf" in lines[0]
    assert "pages" in lines[0] and "chunks" in lines[0]


# --- configuration ---


def test_configure_can_be_called_twice_without_doubling_output():
    logs.configure("INFO")
    logs.configure("INFO")

    assert len(logging.getLogger().handlers) == 1


def test_noisy_libraries_are_quieted():
    logs.configure("DEBUG")

    for name in ("httpx", "openai", "pdfminer"):
        assert logging.getLogger(name).level == logging.WARNING, name
