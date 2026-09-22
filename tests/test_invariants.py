"""The two properties the whole design rests on.

Both are checked end to end, through the real pipeline, because either one
failing would be invisible in a demo and fatal in use.
"""

import json

import pytest

from app.indexer import index_pdf
from app.pipeline import answer_question
from app.schemas import ChatRequest
from tests.conftest import FakeEmbedder, FakeLLM, make_pdf

TOPICS = [
    "eligibility and the maximum amount allowed for each applicant",
    "international applicants and certified translations of documents",
    "the appeals process and the deadline for lodging an appeal",
    "fees payable on submission and the available payment methods",
    "data retention and how long records are kept after a decision",
    "who is responsible for reviewing and approving each application",
]


@pytest.fixture
def indexed(store, settings):
    """A document with comfortably more chunks than we retrieve per query."""
    pages = [
        [("Section {}".format(i + 1), 18)]
        + ["Paragraph {} covers {}.".format(n, topic) for n in range(8)]
        for i, topic in enumerate(TOPICS)
    ]
    embedder = FakeEmbedder()
    index_pdf(make_pdf(pages, title="Handbook"), "handbook.pdf", store, embedder, settings)
    assert store.chunk_count > settings.retriever_top_k
    return embedder


def test_the_model_only_ever_sees_retrieved_chunks_never_the_document(
    store, settings, indexed
):
    """The brief's first rule: do not pass the whole PDF to the LLM."""
    embedder = indexed
    llm = FakeLLM(json.dumps({"answer": "Fifty units.", "answerable": True, "citations": [1]}))

    answer_question(
        ChatRequest(question="What is the maximum amount allowed?", debug=True),
        store,
        embedder,
        llm,
        FakeLLM(),
        settings,
    )

    prompt = llm.calls[0]["user"]
    quoted = [chunk.text for chunk in store.chunks if chunk.text in prompt]

    assert 0 < len(quoted) <= settings.retriever_top_k
    assert len(quoted) < store.chunk_count, "the whole document reached the model"
    assert len(prompt) // 4 <= settings.max_context_tokens * 1.5


def test_every_citation_resolves_to_a_chunk_that_was_retrieved(
    store, settings, indexed
):
    """The model returns evidence positions, never page numbers of its own."""
    embedder = indexed
    llm = FakeLLM(
        json.dumps(
            # 1 is real, 99 is invented and must not reach the user.
            {"answer": "Fifty units.", "answerable": True, "citations": [1, 99]}
        )
    )

    response = answer_question(
        ChatRequest(question="What is the maximum amount allowed?", debug=True),
        store,
        embedder,
        llm,
        FakeLLM(),
        settings,
    )

    used_pages = {
        (item.page_start, item.page_end)
        for item in response.trace.retrieved
        if item.used_as_evidence
    }
    assert response.citations
    for citation in response.citations:
        assert (citation.page_start, citation.page_end) in used_pages
        assert any(
            chunk.page_start == citation.page_start and chunk.section == citation.section
            for chunk in store.chunks
        )
