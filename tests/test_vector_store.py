"""Vector store: search, filtering, persistence and index consistency."""

import pytest

from app.errors import DocumentNotFound
from app.schemas import DocumentInfo
from app.vector_store import VectorStore
from tests.conftest import FakeEmbedder, make_chunk


def _document(doc_id: str, chunks: int) -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id,
        filename=doc_id + ".pdf",
        title=doc_id,
        pages=1,
        chunks=chunks,
        ingested_at="2026-01-01T00:00:00+00:00",
    )


def _add(store: VectorStore, embedder: FakeEmbedder, doc_id: str, texts: list[str]) -> None:
    chunks = [make_chunk(i, text, doc_id=doc_id) for i, text in enumerate(texts)]
    store.add(_document(doc_id, len(chunks)), chunks, embedder.embed(texts))


def test_search_ranks_the_closest_chunk_first(store, embedder):
    _add(
        store,
        embedder,
        "doc1",
        [
            "the maximum amount allowed is fifty units",
            "applications are reviewed by the admissions office",
            "bicycles must be parked in the rear courtyard",
        ],
    )

    results = store.search(embedder.embed(["what is the maximum amount allowed"])[0], top_k=3)

    assert "maximum amount" in results[0].chunk.text
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_search_can_be_restricted_to_one_document(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta gamma delta"])
    _add(store, embedder, "doc2", ["alpha beta gamma delta"])

    results = store.search(embedder.embed(["alpha beta"])[0], top_k=5, doc_id="doc2")

    assert results and all(r.chunk.doc_id == "doc2" for r in results)


def test_search_across_all_documents_by_default(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta gamma delta"])
    _add(store, embedder, "doc2", ["epsilon zeta eta theta"])

    results = store.search(embedder.embed(["alpha epsilon"])[0], top_k=5)

    assert {r.chunk.doc_id for r in results} == {"doc1", "doc2"}


def test_search_for_an_unknown_document_is_an_error(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta"])

    with pytest.raises(DocumentNotFound):
        store.search(embedder.embed(["alpha"])[0], top_k=3, doc_id="nope")


def test_search_on_an_empty_index_returns_nothing(store, embedder):
    assert store.search(embedder.embed(["anything"])[0], top_k=3) == []


def test_the_index_survives_a_restart(settings, embedder):
    first = VectorStore(settings.data_dir)
    _add(first, embedder, "doc1", ["the maximum amount allowed is fifty units"])

    reopened = VectorStore(settings.data_dir)
    reopened.load()

    assert reopened.chunk_count == 1
    assert reopened.has_document("doc1")
    assert reopened.search(embedder.embed(["maximum amount"])[0], top_k=1)[0].chunk.text


def test_loading_a_missing_index_is_not_an_error(settings):
    store = VectorStore(settings.data_dir / "does-not-exist")
    store.load()

    assert store.chunk_count == 0


def test_a_changed_embedding_dimension_is_rejected(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta"])

    with pytest.raises(RuntimeError, match="dimension"):
        _add(store, FakeEmbedder(dim=32), "doc2", ["gamma delta"])
