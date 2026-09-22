"""Hybrid search, reciprocal rank fusion, and the LLM reranker."""

import json

from app.rerank import rerank
from app.retrieval import fuse, retrieve
from tests.conftest import FakeEmbedder, FakeLLM, make_chunk, retrieved


def index(store, embedder, texts: list[str], doc_id: str = "doc1") -> None:
    from app.schemas import Chunk, DocumentInfo

    chunks = [
        Chunk(chunk_id="{}:{}".format(doc_id, i), doc_id=doc_id, doc_title="Doc",
              text=text, page_start=i + 1, page_end=i + 1)
        for i, text in enumerate(texts)
    ]
    store.add(
        DocumentInfo(doc_id=doc_id, filename=doc_id + ".pdf", title="Doc",
                     pages=len(texts), chunks=len(texts),
                     ingested_at="2026-01-01T00:00:00+00:00"),
        chunks, embedder.embed(texts),
    )


# --- keyword search ---


def test_keyword_search_finds_an_exact_term(store, embedder):
    index(store, embedder, [
        "The applicant must supply a reference number when writing to the office.",
        "Form K-9137B must be attached to every submission without exception.",
        "Decisions are posted within thirty working days of the hearing.",
    ])

    hits = store.keyword_search("K-9137B", top_k=3)

    assert hits and "K-9137B" in hits[0].chunk.text


def test_keyword_search_returns_nothing_for_words_not_in_the_corpus(store, embedder):
    index(store, embedder, ["alpha beta gamma"])

    assert store.keyword_search("zzzz qqqq", top_k=3) == []


def test_keyword_search_respects_the_document_filter(store, embedder):
    index(store, embedder, ["shared keyword appears here"], doc_id="doc1")
    index(store, embedder, ["shared keyword appears here too"], doc_id="doc2")

    hits = store.keyword_search("shared keyword", top_k=5, doc_id="doc2")

    assert hits and all(h.chunk.doc_id == "doc2" for h in hits)


def test_the_keyword_index_shrinks_with_a_deleted_document(store, embedder):
    index(store, embedder, ["quarantine procedures for imported goods"], doc_id="doc1")
    index(store, embedder, ["unrelated text about bicycles"], doc_id="doc2")

    store.delete_document("doc1")

    assert store.keyword_search("quarantine", top_k=5) == []


# --- fusion ---


def test_fusion_ranks_a_chunk_both_rankers_agree_on_first():
    a, b, c = (make_chunk(i, "chunk {}".format(i), page=i + 1) for i in range(3))
    dense = retrieved(a, b, c)       # a first
    keyword = retrieved(c, a, b)     # a second

    fused = fuse([dense, keyword], rrf_k=60, top_k=3)

    assert fused[0].chunk.chunk_id == a.chunk_id


def test_fusion_keeps_a_chunk_only_one_ranker_found():
    only_dense = make_chunk(0, "found by meaning", page=1)
    only_keyword = make_chunk(1, "found by exact term", page=2)

    fused = fuse([retrieved(only_dense), retrieved(only_keyword)], rrf_k=60, top_k=5)

    assert {f.chunk.chunk_id for f in fused} == {only_dense.chunk_id, only_keyword.chunk_id}


def test_fusion_uses_rank_not_score():
    # The two rankers' scores are on different scales; only the order may count.
    high = make_chunk(0, "high scoring but second", page=1)
    low = make_chunk(1, "low scoring but first in both", page=2)
    from app.schemas import RetrievedChunk

    dense = [RetrievedChunk(chunk=low, score=0.01), RetrievedChunk(chunk=high, score=0.99)]
    keyword = [RetrievedChunk(chunk=low, score=0.02), RetrievedChunk(chunk=high, score=50.0)]

    assert fuse([dense, keyword], rrf_k=60, top_k=2)[0].chunk.chunk_id == low.chunk_id


# --- hybrid retrieval end to end ---


def test_hybrid_finds_an_exact_term_that_dense_search_ranks_lower(store, embedder):
    index(store, embedder, [
        "general guidance about submitting paperwork to the department",
        "guidance on paperwork and departmental submission procedures",
        "Form K-9137B is the only accepted attachment.",
    ])

    dense_only = retrieve(store, embedder, "K-9137B", top_k=1, hybrid=False)
    hybrid = retrieve(store, embedder, "K-9137B", top_k=1, hybrid=True)

    assert "K-9137B" in hybrid[0].chunk.text
    assert dense_only != hybrid or "K-9137B" in dense_only[0].chunk.text


def test_hybrid_falls_back_to_dense_when_no_keyword_matches(store, embedder):
    index(store, embedder, ["alpha beta gamma", "delta epsilon zeta"])

    hits = retrieve(store, embedder, "qqqq zzzz", top_k=2, hybrid=True)

    assert len(hits) == 2, "a query with no keyword hits still returns dense results"


def test_retrieval_is_unchanged_when_hybrid_is_off(store, embedder):
    index(store, embedder, ["alpha beta gamma", "delta epsilon zeta"])

    assert retrieve(store, embedder, "alpha", top_k=2, hybrid=False) == store.search(
        embedder.embed(["alpha"])[0], top_k=2
    )


# --- reranking ---


def order(*positions) -> str:
    return json.dumps({"order": list(positions)})


def candidates(n: int):
    return retrieved(*[make_chunk(i, "passage {}".format(i), page=i + 1) for i in range(n)])


def test_reranking_reorders_to_what_the_model_chose():
    items = candidates(4)

    chosen, guards = rerank(FakeLLM(order(3, 1)), "question", items, top_k=2)

    assert [c.chunk.chunk_id for c in chosen] == [
        items[2].chunk.chunk_id, items[0].chunk.chunk_id
    ]
    assert guards == []


def test_reranking_tops_up_when_the_model_returns_too_few():
    items = candidates(4)

    chosen, _ = rerank(FakeLLM(order(4)), "question", items, top_k=3)

    assert chosen[0].chunk.chunk_id == items[3].chunk.chunk_id
    assert len(chosen) == 3, "the model must not be able to starve the answer of evidence"


def test_an_invented_position_is_dropped_and_recorded():
    items = candidates(3)

    chosen, guards = rerank(FakeLLM(order(2, 99)), "question", items, top_k=3)

    assert all(c.chunk.chunk_id in {i.chunk.chunk_id for i in items} for c in chosen)
    assert "rerank_dropped:99" in guards


def test_a_failing_reranker_leaves_the_original_order_alone():
    items = candidates(3)

    chosen, guards = rerank(FakeLLM("not json at all"), "question", items, top_k=2)

    assert [c.chunk.chunk_id for c in chosen] == [i.chunk.chunk_id for i in items[:2]]
    assert guards == ["rerank_failed"]


def test_a_reranker_that_errors_does_not_bring_the_answer_down():
    items = candidates(3)

    chosen, guards = rerank(FakeLLM(RuntimeError("model down")), "q", items, top_k=2)

    assert len(chosen) == 2
    assert guards == ["rerank_failed"]


def test_a_single_candidate_is_not_worth_a_model_call():
    llm = FakeLLM()  # scripted with nothing: calling it would fail the test

    chosen, guards = rerank(llm, "question", candidates(1), top_k=3)

    assert len(chosen) == 1
    assert llm.calls == []
