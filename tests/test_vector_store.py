"""Vector store: search, filtering, persistence and index consistency."""

import pytest

from app.db import Database
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
    first = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    first.load()
    _add(first, embedder, "doc1", ["the maximum amount allowed is fifty units"])

    reopened = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    reopened.load()

    assert reopened.chunk_count == 1
    assert reopened.has_document("doc1")
    assert reopened.search(embedder.embed(["maximum amount"])[0], top_k=1)[0].chunk.text


def test_loading_an_empty_database_is_not_an_error(settings):
    store = VectorStore(Database(settings.data_dir / "fresh.db"), settings.data_dir)
    store.load()

    assert store.chunk_count == 0


def test_a_changed_embedding_dimension_is_rejected(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta"])

    with pytest.raises(RuntimeError, match="dimension"):
        _add(store, FakeEmbedder(dim=32), "doc2", ["gamma delta"])


# --- deletion, which the file-based index could not do at all ---


def test_deleting_a_document_removes_its_chunks_and_leaves_the_rest(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta gamma", "alpha beta delta"])
    _add(store, embedder, "doc2", ["epsilon zeta eta"])

    store.delete_document("doc1")

    assert not store.has_document("doc1")
    assert store.chunk_count == 1
    assert store.embeddings.shape[0] == 1, "the matrix must shrink with the chunks"
    assert store.search(embedder.embed(["epsilon"])[0], top_k=5)[0].chunk.doc_id == "doc2"


def test_deletion_survives_a_restart(settings, embedder):
    store = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    store.load()
    _add(store, embedder, "doc1", ["alpha beta"])
    _add(store, embedder, "doc2", ["gamma delta"])
    store.delete_document("doc1")

    reopened = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    reopened.load()

    assert [d.doc_id for d in reopened.list_documents()] == ["doc2"]
    assert reopened.chunk_count == 1


def test_deleting_an_unknown_document_is_an_error(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta"])

    with pytest.raises(DocumentNotFound):
        store.delete_document("nope")


def test_the_index_is_searchable_again_after_deleting_everything(store, embedder):
    _add(store, embedder, "doc1", ["alpha beta"])
    store.delete_document("doc1")

    assert store.chunk_count == 0
    assert store.search(embedder.embed(["alpha"])[0], top_k=3) == []


# --- migration from the old two-file index ---


def test_an_index_written_as_files_is_carried_into_the_database(settings, embedder):
    """The old layout kept no PDFs, so its vectors cannot be rebuilt - only moved."""
    import json

    import numpy as np

    from app.schemas import Chunk

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    chunk = Chunk(
        chunk_id="old:00000", doc_id="doc1", doc_title="Legacy",
        text="the maximum amount allowed is fifty units",
        section="Eligibility", page_start=7, page_end=7,
    )
    (settings.data_dir / "chunks.jsonl").write_text(
        chunk.model_dump_json() + "\n", encoding="utf-8"
    )
    with (settings.data_dir / "embeddings.npy").open("wb") as handle:
        np.save(handle, np.asarray(embedder.embed([chunk.text]), dtype=np.float32))
    (settings.data_dir / "documents.json").write_text(
        json.dumps({"doc1": _document("doc1", 1).model_dump()}), encoding="utf-8"
    )

    store = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    store.load()

    assert store.chunk_count == 1
    assert store.chunks[0].section == "Eligibility"
    assert store.chunks[0].page_start == 7
    assert store.search(embedder.embed(["maximum amount"])[0], top_k=1)[0].score > 0.5
    assert not (settings.data_dir / "chunks.jsonl").exists(), "old files should be set aside"
    assert (settings.data_dir / "chunks.jsonl.migrated").exists()
