"""The activity record behind the dashboard."""

import json

from fastapi.testclient import TestClient

from app import activity
from app.main import create_app
from tests.conftest import FakeEmbedder, FakeLLM
from tests.test_api import FakeProviders, answer


def loaded(settings, store, simple_pdf, llm=None):
    providers = FakeProviders(FakeEmbedder(), llm=llm)
    client = TestClient(create_app(settings, store, providers))
    client.__enter__()
    client.post("/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")})
    return client, providers


def test_nothing_asked_means_an_empty_dashboard(settings, store):
    with TestClient(create_app(settings, store, FakeProviders(FakeEmbedder()))) as client:
        body = client.get("/activity").json()

    assert body["stats"]["questions"] == 0
    assert body["recent"] == []


def test_a_question_is_recorded_with_its_outcome(settings, store, simple_pdf):
    client, _ = loaded(settings, store, simple_pdf,
                       llm=FakeLLM(answer("The maximum is 50 units.")))

    client.post("/chat", json={"question": "What is the maximum amount allowed?"})
    body = client.get("/activity").json()

    assert body["stats"]["questions"] == 1
    assert body["stats"]["answered"] == 1
    assert body["stats"]["answered_share"] == 1.0
    entry = body["recent"][0]
    assert entry["question"] == "What is the maximum amount allowed?"
    assert entry["answerable"] is True
    assert entry["citations"] == 1
    assert entry["latency_ms"] >= 0
    client.__exit__(None, None, None)


def test_a_refusal_is_counted_separately(settings, store, simple_pdf):
    client, _ = loaded(settings, store, simple_pdf,
                       llm=FakeLLM(answer("Not in here.", answerable=False, citations=[])))

    client.post("/chat", json={"question": "Where do I park?"})
    stats = client.get("/activity").json()["stats"]

    assert (stats["answered"], stats["refused"]) == (0, 1)
    assert stats["answered_share"] == 0.0
    client.__exit__(None, None, None)


def test_guards_are_counted_so_a_pattern_is_visible(settings, store, simple_pdf):
    # Two questions where the model had to be retried before it made sense.
    client, providers = loaded(settings, store, simple_pdf)
    for _ in range(2):
        providers.llm.responses.extend(["not json", answer("Fifty units.")])
        client.post("/chat", json={"question": "What is the maximum?"})

    guards = client.get("/activity").json()["stats"]["guards"]

    assert guards.get("json_retry") == 2
    client.__exit__(None, None, None)


def test_the_newest_question_comes_first(settings, store, simple_pdf):
    client, providers = loaded(settings, store, simple_pdf)
    for n in range(3):
        providers.llm.responses.append(answer("Answer {}".format(n)))
        client.post("/chat", json={"question": "Question {}".format(n)})

    recent = client.get("/activity").json()["recent"]

    assert [r["question"] for r in recent] == ["Question 2", "Question 1", "Question 0"]
    client.__exit__(None, None, None)


def test_the_record_survives_a_restart(settings, store, simple_pdf):
    client, _ = loaded(settings, store, simple_pdf, llm=FakeLLM(answer("Fifty units.")))
    client.post("/chat", json={"question": "What is the maximum?"})
    client.__exit__(None, None, None)

    with TestClient(create_app(settings, store, FakeProviders(FakeEmbedder()))) as again:
        assert again.get("/activity").json()["stats"]["questions"] == 1


def test_an_evaluation_run_does_not_fill_the_dashboard(settings, store, simple_pdf):
    """The dashboard is for questions people asked, not for a benchmark."""
    providers = FakeProviders(
        FakeEmbedder(),
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(120)]),
        rewrite_llm=FakeLLM(*["standalone" for _ in range(40)]),
    )
    with TestClient(create_app(settings, store, providers)) as client:
        doc_id = client.post(
            "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
        ).json()["doc_id"]
        client.post("/eval/run", json={"doc_id": doc_id, "judge": False})
        from tests.test_eval_api import wait_for_finish

        wait_for_finish(client)

        assert client.get("/activity").json()["stats"]["questions"] == 0


def test_a_broken_record_does_not_break_the_answer(settings, store, simple_pdf, monkeypatch):
    client, _ = loaded(settings, store, simple_pdf, llm=FakeLLM(answer("Fifty units.")))

    def explode(*args, **kwargs):
        raise RuntimeError("disk is full")

    monkeypatch.setattr(activity.Database, "write", explode)
    response = client.post("/chat", json={"question": "What is the maximum?"})

    assert response.status_code == 200, "observability must never cost an answer"
    assert response.json()["answerable"] is True
    client.__exit__(None, None, None)
