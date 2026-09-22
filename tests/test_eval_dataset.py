"""The evaluation dataset is a deliverable, so it is checked like code.

These run offline and check shape, not answers: that the file parses, covers
the categories the brief asks for, and that every question is one the API would
actually accept. Whether the gold pages are right is verified against the
document itself when the set is written.
"""

import json
from pathlib import Path

import pytest

from app.schemas import ChatRequest

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
REQUIRED_TYPES = {
    "factual",
    "multi_passage",
    "follow_up",
    "unanswerable",
    "similar_sections",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_set_covers_every_category_evenly(rows):
    counts = {name: sum(1 for r in rows if r["type"] == name) for name in REQUIRED_TYPES}

    assert 15 <= len(rows) <= 20, "the brief asks for roughly 15 to 20"
    assert len(set(counts.values())) == 1, "the same number in each category"


def test_every_category_is_covered(rows):
    assert {row["type"] for row in rows} == REQUIRED_TYPES


def test_question_ids_are_unique(rows):
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_answerable_questions_carry_gold_pages(rows):
    for row in rows:
        if row["answerable"]:
            assert row["gold_pages"], "{} needs gold pages to score retrieval".format(row["id"])
            assert all(isinstance(page, int) and page > 0 for page in row["gold_pages"])


def test_unanswerable_questions_claim_no_evidence(rows):
    unanswerable = [row for row in rows if not row["answerable"]]

    assert unanswerable, "the set must test refusal"
    for row in unanswerable:
        assert row["gold_pages"] == []


def test_every_question_names_the_document_it_belongs_to(rows):
    # Several unanswerable questions ARE answerable from the other document, so
    # scoping each question to its own is what makes them an abstention test.
    for row in rows:
        assert row["doc"], "{} does not name a document".format(row["id"])


def test_every_question_carries_a_reference_answer(rows):
    for row in rows:
        assert row["expected_answer"].strip(), "{} has nothing to judge against".format(row["id"])


def test_every_question_lists_the_points_a_full_answer_covers(rows):
    # The judge grades against these as well as the prose, which keeps scoring
    # steadier between runs.
    for row in rows:
        assert row["expected_points"], "{} lists no expected points".format(row["id"])


def test_the_unanswerable_questions_are_split_across_both_documents(rows):
    docs = {row["doc"] for row in rows if not row["answerable"]}

    assert len(docs) > 1, "abstention must be tested in both directions"


def test_follow_ups_actually_depend_on_history(rows):
    follow_ups = [row for row in rows if row["type"] == "follow_up"]

    assert follow_ups
    for row in follow_ups:
        assert row["history"], "{} is not a follow-up without history".format(row["id"])


def test_every_question_is_one_the_api_would_accept(rows):
    # Catches an over-long question or a bad history role before it wastes a
    # whole evaluation run.
    for row in rows:
        ChatRequest(question=row["question"], history=row["history"])


# --- matching the set to indexed documents ---


def indexed(store, embedder, doc_id: str, filename: str, title: str) -> None:
    from app.schemas import Chunk, DocumentInfo

    chunk = Chunk(chunk_id=doc_id + ":0", doc_id=doc_id, doc_title=title,
                  text="some indexed text", page_start=1, page_end=1)
    store.add(
        DocumentInfo(doc_id=doc_id, filename=filename, title=title, pages=1, chunks=1,
                     ingested_at="2026-01-01T00:00:00+00:00"),
        [chunk], embedder.embed([chunk.text]),
    )


def question(doc: str) -> dict:
    return {"id": "q", "doc": doc, "type": "factual", "question": "?",
            "answerable": True, "gold_pages": [1]}


def test_a_document_is_matched_by_content_even_when_renamed(store, embedder, tmp_path):
    # The uploader can call the file anything; the bytes are what identify it.
    import hashlib

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "free221.pdf").write_bytes(b"%PDF-1.4 pretend")
    digest = hashlib.sha256(b"%PDF-1.4 pretend").hexdigest()
    indexed(store, embedder, digest, "calculus.pdf", "calculus")

    from app.evaluation import resolve_documents

    assert resolve_documents([question("free221")], store, corpus) == {"free221": digest}


def test_a_document_is_matched_by_filename_when_the_corpus_file_is_absent(
    store, embedder, tmp_path
):
    from app.evaluation import resolve_documents

    indexed(store, embedder, "abc123", "free221.pdf", "Some Title")

    assert resolve_documents([question("free221")], store, tmp_path) == {"free221": "abc123"}


def test_a_document_is_matched_by_title_as_a_last_resort(store, embedder, tmp_path):
    from app.evaluation import resolve_documents

    indexed(store, embedder, "abc123", "upload-4.pdf", "Lecture10")

    assert resolve_documents([question("Lecture10")], store, tmp_path) == {"Lecture10": "abc123"}


def test_an_unmatched_document_says_what_is_indexed(store, embedder, tmp_path):
    """The old message named only what was missing, which left you guessing."""
    from app.errors import IngestError
    from app.evaluation import resolve_documents

    indexed(store, embedder, "abc123", "something-else.pdf", "Something Else")

    with pytest.raises(IngestError) as caught:
        resolve_documents([question("free221")], store, tmp_path)

    assert "free221" in str(caught.value)
    assert "something-else.pdf" in str(caught.value), "it must name what is indexed"
