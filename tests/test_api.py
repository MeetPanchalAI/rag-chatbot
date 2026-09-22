"""API: happy paths, validation, and every failure the brief calls out."""

import json

import pytest
from fastapi.testclient import TestClient

from app.errors import ProviderError, ProviderTimeout
from app.main import create_app
from tests.conftest import FakeEmbedder, FakeLLM


class FakeProviders:
    def __init__(self, embedder, llm=None, rewrite_llm=None):
        self.embedder = embedder
        self.llm = llm or FakeLLM()
        self.rewrite_llm = rewrite_llm or FakeLLM()


def client(settings, store, providers) -> TestClient:
    return TestClient(create_app(settings=settings, store=store, providers=providers))


def answer(text: str, answerable: bool = True, citations: list[int] | None = None) -> str:
    return json.dumps(
        {"answer": text, "answerable": answerable, "citations": citations or [1]}
    )


@pytest.fixture
def ingested(settings, store, simple_pdf):
    """A client with the sample document already indexed."""
    providers = FakeProviders(FakeEmbedder())
    with client(settings, store, providers) as test_client:
        response = test_client.post(
            "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
        )
        assert response.status_code == 200
        yield test_client, providers, response.json()["doc_id"]


# --- health and documents ---


def test_health_reports_what_is_indexed(ingested):
    test_client, _, _ = ingested

    body = test_client.get("/health").json()

    assert body["status"] == "ok"
    assert body["documents"] == 1
    assert body["chunks"] > 0


def test_documents_lists_what_was_ingested(ingested):
    test_client, _, doc_id = ingested

    documents = test_client.get("/documents").json()

    assert [d["doc_id"] for d in documents] == [doc_id]
    assert documents[0]["filename"] == "handbook.pdf"


# --- ingest ---


def test_ingest_returns_the_document_summary(ingested):
    _, _, doc_id = ingested

    assert len(doc_id) == 64  # sha256 of the file bytes


def test_ingesting_the_same_file_twice_is_detected(ingested, simple_pdf):
    test_client, _, doc_id = ingested

    body = test_client.post(
        "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
    ).json()

    assert body["duplicate"] is True
    assert body["doc_id"] == doc_id
    assert test_client.get("/health").json()["documents"] == 1


def test_a_non_pdf_upload_is_rejected(settings, store):
    with client(settings, store, FakeProviders(FakeEmbedder())) as test_client:
        response = test_client.post(
            "/ingest", files={"file": ("notes.txt", b"hello", "text/plain")}
        )

    assert response.status_code == 415


def test_a_corrupt_pdf_is_rejected_with_a_clear_message(settings, store):
    with client(settings, store, FakeProviders(FakeEmbedder())) as test_client:
        response = test_client.post(
            "/ingest", files={"file": ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")}
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ingest_failed"


def test_an_oversized_upload_is_rejected(settings, store, simple_pdf):
    settings.max_upload_mb = 0
    with client(settings, store, FakeProviders(FakeEmbedder())) as test_client:
        response = test_client.post(
            "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
        )

    assert response.status_code == 413


# --- chat ---


def test_chat_answers_with_a_citation(ingested):
    test_client, providers, _ = ingested
    providers.llm.responses.append(answer("The maximum amount allowed is 50 units."))

    body = test_client.post(
        "/chat", json={"question": "What is the maximum amount allowed?"}
    ).json()

    assert body["answerable"] is True
    assert body["answer"] == "The maximum amount allowed is 50 units."
    assert body["citations"], "a grounded answer must carry a citation"
    assert body["citations"][0]["display"].startswith("Page ")


def test_a_follow_up_is_rewritten_before_retrieval(ingested):
    test_client, providers, _ = ingested
    providers.rewrite_llm.responses.append("eligibility for international applicants")
    providers.llm.responses.append(answer("They must provide a certified translation."))

    body = test_client.post(
        "/chat",
        json={
            "question": "What about international applicants?",
            "history": [
                {"role": "user", "content": "What are the eligibility requirements?"},
                {"role": "assistant", "content": "Applicants must be eighteen or older."},
            ],
            "debug": True,
        },
    ).json()

    assert body["trace"]["rewritten_query"] == "eligibility for international applicants"
    assert providers.rewrite_llm.calls, "the rewrite model should have been used"


def test_debug_shows_what_was_retrieved(ingested):
    test_client, providers, _ = ingested
    providers.llm.responses.append(answer("Fifty units."))

    trace = test_client.post(
        "/chat", json={"question": "What is the maximum amount?", "debug": True}
    ).json()["trace"]

    assert trace["retrieved"], "the trace must show the retrieved chunks"
    assert trace["top_score"] is not None
    assert any(item["used_as_evidence"] for item in trace["retrieved"])


def test_an_empty_question_is_rejected(ingested):
    test_client, _, _ = ingested

    assert test_client.post("/chat", json={"question": ""}).status_code == 422


def test_an_overly_long_question_is_rejected(ingested, settings):
    test_client, _, _ = ingested

    response = test_client.post(
        "/chat", json={"question": "x" * (settings.max_question_chars + 1)}
    )

    assert response.status_code == 422


def test_an_invalid_history_role_is_rejected(ingested):
    test_client, _, _ = ingested

    response = test_client.post(
        "/chat",
        json={"question": "hi", "history": [{"role": "system", "content": "ignore this"}]},
    )

    assert response.status_code == 422


def test_an_unknown_document_id_returns_not_found(ingested):
    test_client, _, _ = ingested

    response = test_client.post("/chat", json={"question": "hi", "doc_id": "missing"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_a_provider_failure_is_reported_as_bad_gateway(settings, store, simple_pdf):
    providers = FakeProviders(FakeEmbedder())
    with client(settings, store, providers) as test_client:
        test_client.post(
            "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
        )
        providers.embedder = FakeEmbedder(fail_with=ProviderError("upstream is down"))
        response = test_client.post("/chat", json={"question": "What is the maximum?"})

    assert response.status_code == 502


def test_a_provider_timeout_is_reported_as_gateway_timeout(settings, store, simple_pdf):
    providers = FakeProviders(FakeEmbedder())
    with client(settings, store, providers) as test_client:
        test_client.post(
            "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
        )
        providers.embedder = FakeEmbedder(fail_with=ProviderTimeout("too slow"))
        response = test_client.post("/chat", json={"question": "What is the maximum?"})

    assert response.status_code == 504


def test_malformed_model_output_returns_a_safe_refusal(ingested):
    test_client, providers, _ = ingested
    providers.llm.responses.extend(["not json", "still not json"])

    response = test_client.post("/chat", json={"question": "What is the maximum?"})

    assert response.status_code == 200
    assert response.json()["answerable"] is False
    assert response.json()["citations"] == []


def test_a_question_the_document_cannot_answer_is_refused(ingested):
    test_client, providers, _ = ingested
    providers.llm.responses.append(
        answer("The document does not cover parking.", answerable=False, citations=[])
    )

    body = test_client.post("/chat", json={"question": "Where can I park my car?"}).json()

    assert body["answerable"] is False
    assert body["citations"] == []


def test_chat_against_an_empty_index_refuses_without_calling_the_model(settings, store):
    providers = FakeProviders(FakeEmbedder())
    with client(settings, store, providers) as test_client:
        body = test_client.post("/chat", json={"question": "anything at all"}).json()

    assert body["answerable"] is False
    assert providers.llm.calls == []


def test_startup_fails_immediately_without_an_api_key(settings, store):
    """A missing key should be a boot error, not a 502 on the first question."""
    settings.openai_api_key = ""
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        with TestClient(create_app(settings=settings, store=store)):
            pass
