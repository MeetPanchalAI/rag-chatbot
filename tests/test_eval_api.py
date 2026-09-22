"""The browser UI route and the evaluation endpoints."""

import time

from fastapi.testclient import TestClient

from app.evaluation import load_questions
from app.main import create_app
from app.schemas import EvalStatus
from tests.conftest import FakeEmbedder, FakeLLM
from tests.test_api import FakeProviders, answer


def wait_for_finish(client: TestClient, timeout: float = 10.0) -> dict:
    """Poll the status endpoint the way the UI does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/eval/status").json()
        if not status["running"]:
            return status
        time.sleep(0.02)
    raise AssertionError("the evaluation did not finish in time")


def test_the_ui_is_served_at_the_root(settings, store):
    providers = FakeProviders(FakeEmbedder())
    with TestClient(create_app(settings, store, providers)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Document Chatbot" in response.text


def test_the_ui_only_uses_endpoints_that_exist(settings, store):
    """The page is hand-written, so a renamed endpoint would break it silently."""
    providers = FakeProviders(FakeEmbedder())
    with TestClient(create_app(settings, store, providers)) as client:
        page = client.get("/").text
        paths = set(client.get("/openapi.json").json()["paths"])

    for path in ["/health", "/documents", "/ingest", "/chat", "/eval/run", "/eval/status"]:
        assert "'{}'".format(path) in page, "{} is not called by the UI".format(path)
        assert path in paths, "{} is called by the UI but not served".format(path)


def test_status_is_idle_before_any_run(settings, store):
    providers = FakeProviders(FakeEmbedder())
    with TestClient(create_app(settings, store, providers)) as client:
        status = client.get("/eval/status").json()

    assert status == EvalStatus().model_dump()


def test_running_the_evaluation_scores_every_question(settings, store, simple_pdf):
    providers = FakeProviders(
        FakeEmbedder(),
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(60)]),
        rewrite_llm=FakeLLM(*["standalone query" for _ in range(20)]),
    )
    expected = len(load_questions())

    with TestClient(create_app(settings, store, providers)) as client:
        client.post("/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")})
        started = client.post("/eval/run", json={})
        assert started.status_code == 200
        assert started.json()["total"] == expected

        status = wait_for_finish(client)

    assert status["error"] is None
    assert status["done"] == expected
    assert len(status["rows"]) == expected
    assert "Retrieval recall (any gold page found)" in status["summary"]
    assert {row["id"] for row in status["rows"]} == {q["id"] for q in load_questions()}


def test_the_report_is_written_where_configured(settings, store, simple_pdf):
    providers = FakeProviders(
        FakeEmbedder(),
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(60)]),
        rewrite_llm=FakeLLM(*["standalone query" for _ in range(20)]),
    )

    with TestClient(create_app(settings, store, providers)) as client:
        client.post("/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")})
        client.post("/eval/run", json={})
        wait_for_finish(client)

    report = (settings.eval_dir / "results.md").read_text(encoding="utf-8")
    assert "# Evaluation results" in report
    assert (settings.eval_dir / "results.jsonl").exists()


def test_a_second_run_is_refused_while_one_is_in_flight(settings, store):
    providers = FakeProviders(FakeEmbedder())
    app = create_app(settings, store, providers)

    with TestClient(app) as client:
        app.state.eval_run.status = EvalStatus(running=True, done=2, total=18)
        response = client.post("/eval/run", json={})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "evaluation_running"


def test_evaluating_an_unknown_document_is_refused(settings, store, simple_pdf):
    providers = FakeProviders(FakeEmbedder())

    with TestClient(create_app(settings, store, providers)) as client:
        client.post("/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")})
        response = client.post("/eval/run", json={"doc_id": "not-a-real-document"})

    assert response.status_code == 404


def test_a_provider_failure_is_reported_rather_than_lost(settings, store, simple_pdf):
    # The run happens on a worker thread; a failure there must still reach the UI.
    providers = FakeProviders(FakeEmbedder())

    with TestClient(create_app(settings, store, providers)) as client:
        client.post("/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")})
        providers.embedder = FakeEmbedder(fail_with=RuntimeError("upstream is down"))
        client.post("/eval/run", json={})
        status = wait_for_finish(client)

    assert status["error"] is not None
    assert "upstream is down" in status["error"]
