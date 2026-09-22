"""What the database buys: deletable documents, stored conversations, run history."""

import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeEmbedder, FakeLLM
from tests.test_api import FakeProviders, answer
from tests.test_eval_api import wait_for_finish


def loaded(settings, store, simple_pdf, llm=None, rewrite_llm=None):
    providers = FakeProviders(FakeEmbedder(), llm=llm, rewrite_llm=rewrite_llm)
    client = TestClient(create_app(settings, store, providers))
    client.__enter__()
    doc_id = client.post(
        "/ingest", files={"file": ("handbook.pdf", simple_pdf, "application/pdf")}
    ).json()["doc_id"]
    return client, providers, doc_id


# --- documents ---


def test_the_source_pdf_is_kept_so_the_corpus_can_be_rebuilt(settings, store, simple_pdf):
    client, _, doc_id = loaded(settings, store, simple_pdf)

    kept = store.source_path(doc_id)

    assert kept is not None and kept.exists()
    assert kept.read_bytes() == simple_pdf, "the stored PDF must be the file we were given"
    client.__exit__(None, None, None)


def test_a_document_can_be_deleted_and_stops_being_searchable(settings, store, simple_pdf):
    client, providers, doc_id = loaded(settings, store, simple_pdf)
    kept = store.source_path(doc_id)

    response = client.delete("/documents/" + doc_id)

    assert response.status_code == 200
    assert client.get("/health").json() == {"status": "ok", "documents": 0, "chunks": 0}
    assert client.get("/documents").json() == []
    assert not kept.exists(), "the stored PDF should go with the document"
    client.__exit__(None, None, None)


def test_deleting_a_document_that_is_not_there_is_a_404(settings, store, simple_pdf):
    client, _, _ = loaded(settings, store, simple_pdf)

    assert client.delete("/documents/nope").status_code == 404
    client.__exit__(None, None, None)


# --- conversations ---


def test_a_conversation_records_both_turns_and_survives_a_restart(
    settings, store, simple_pdf
):
    llm = FakeLLM(answer("The maximum amount allowed is 50 units."))
    client, _, _ = loaded(settings, store, simple_pdf, llm=llm)

    conversation = client.post("/conversations", json={}).json()
    client.post("/chat", json={
        "question": "What is the maximum amount allowed?",
        "conversation_id": conversation["id"],
    })
    client.__exit__(None, None, None)

    # A fresh app over the same database.
    again = TestClient(create_app(settings, store, FakeProviders(FakeEmbedder())))
    with again:
        stored = again.get("/conversations/" + conversation["id"]).json()

    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][1]["answerable"] is True
    assert stored["messages"][1]["citations"], "citations are kept with the answer"
    assert stored["title"].startswith("What is the maximum"), "named after the first question"


def test_the_server_supplies_the_history_for_a_stored_conversation(
    settings, store, simple_pdf
):
    llm = FakeLLM(answer("Applicants must be eighteen."), answer("A certified translation."))
    rewrite = FakeLLM("eligibility for international applicants")
    client, providers, _ = loaded(settings, store, simple_pdf, llm=llm, rewrite_llm=rewrite)
    conversation = client.post("/conversations", json={}).json()

    client.post("/chat", json={
        "question": "What are the eligibility requirements?",
        "conversation_id": conversation["id"],
    })
    # No history in this request: the server must find it.
    client.post("/chat", json={
        "question": "What about international applicants?",
        "conversation_id": conversation["id"],
    })

    prompt = providers.rewrite_llm.calls[0]["user"]
    assert "eligibility requirements" in prompt, "prior turns were not supplied"
    client.__exit__(None, None, None)


def test_passing_history_directly_writes_nothing(settings, store, simple_pdf):
    llm = FakeLLM(answer("Fifty units."))
    client, _, _ = loaded(settings, store, simple_pdf, llm=llm)

    client.post("/chat", json={"question": "What is the maximum?", "history": []})

    assert client.get("/conversations").json() == [], "the stateless path must stay stateless"
    client.__exit__(None, None, None)


def test_conversations_can_be_listed_and_deleted(settings, store, simple_pdf):
    client, _, _ = loaded(settings, store, simple_pdf)
    first = client.post("/conversations", json={"title": "Fees"}).json()
    client.post("/conversations", json={"title": "Eligibility"})

    assert {c["title"] for c in client.get("/conversations").json()} == {"Fees", "Eligibility"}
    assert client.delete("/conversations/" + first["id"]).status_code == 200
    assert [c["title"] for c in client.get("/conversations").json()] == ["Eligibility"]
    assert client.get("/conversations/" + first["id"]).status_code == 404
    client.__exit__(None, None, None)


# --- evaluation history ---


def test_every_run_is_kept_with_the_settings_that_produced_it(settings, store, simple_pdf):
    client, _, doc_id = loaded(
        settings, store, simple_pdf,
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(120)]),
        rewrite_llm=FakeLLM(*["standalone" for _ in range(40)]),
    )

    client.post("/eval/run", json={"label": "baseline", "doc_id": doc_id, "judge": False})
    wait_for_finish(client)
    settings.retriever_top_k = 5
    client.post("/eval/run", json={"label": "top_k 5", "doc_id": doc_id, "judge": False})
    wait_for_finish(client)

    runs = client.get("/eval/runs").json()

    assert [r["label"] for r in runs] == ["top_k 5", "baseline"], "newest first"
    assert runs[0]["knobs"]["retriever_top_k"] == "5"
    assert runs[1]["knobs"]["retriever_top_k"] == "3"
    assert all(r["status"] == "done" for r in runs)
    client.__exit__(None, None, None)


def test_an_old_run_can_be_read_back_in_full(settings, store, simple_pdf):
    client, _, doc_id = loaded(
        settings, store, simple_pdf,
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(120)]),
        rewrite_llm=FakeLLM(*["standalone" for _ in range(40)]),
    )
    client.post("/eval/run", json={"doc_id": doc_id, "judge": False})
    live = wait_for_finish(client)

    stored = client.get("/eval/runs/{}".format(live["run_id"])).json()

    assert stored["summary"] == live["summary"]
    assert len(stored["rows"]) == len(live["rows"])
    assert stored["rows"][0]["id"] == live["rows"][0]["id"]
    assert stored["headline"], "the tiles are kept, not recomputed"
    client.__exit__(None, None, None)


def test_a_run_can_be_deleted_and_takes_its_results_with_it(settings, store, simple_pdf):
    client, _, doc_id = loaded(
        settings, store, simple_pdf,
        llm=FakeLLM(*[answer("Fifty units.") for _ in range(120)]),
        rewrite_llm=FakeLLM(*["standalone" for _ in range(40)]),
    )
    client.post("/eval/run", json={"doc_id": doc_id, "judge": False})
    run_id = wait_for_finish(client)["run_id"]

    assert client.delete("/eval/runs/{}".format(run_id)).status_code == 200
    assert client.get("/eval/runs/{}".format(run_id)).status_code == 404
    assert store.db.query("SELECT 1 FROM eval_results WHERE run_id = ?", run_id) == []
    client.__exit__(None, None, None)
