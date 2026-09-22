"""Retrieval: follow-up rewriting and evidence assembly."""

from app.retrieval import build_evidence, rewrite_query
from app.schemas import Message
from app.text_utils import estimate_tokens
from tests.conftest import FakeLLM, make_chunk, retrieved

HISTORY = [
    Message(role="user", content="What are the eligibility requirements?"),
    Message(role="assistant", content="Applicants must be eighteen or older."),
]


def test_a_first_question_is_not_rewritten():
    llm = FakeLLM()  # scripted with nothing: calling it would fail the test

    query, was_rewritten = rewrite_query(llm, "What is the maximum amount?", [], 3)

    assert query == "What is the maximum amount?"
    assert was_rewritten is False
    assert llm.calls == []


def test_a_follow_up_is_rewritten_into_a_standalone_query():
    llm = FakeLLM("What are the eligibility requirements for international applicants?")

    query, was_rewritten = rewrite_query(llm, "What about international applicants?", HISTORY, 3)

    assert query == "What are the eligibility requirements for international applicants?"
    assert was_rewritten is True


def test_history_is_trimmed_to_the_configured_number_of_turns():
    long_history = [
        Message(role="user" if i % 2 == 0 else "assistant", content="turn {}".format(i))
        for i in range(20)
    ]
    llm = FakeLLM("standalone query")

    rewrite_query(llm, "and then?", long_history, max_turns=2)

    prompt = llm.calls[0]["user"]
    assert "turn 19" in prompt
    assert "turn 15" not in prompt


def test_an_empty_rewrite_falls_back_to_the_original_question():
    query, was_rewritten = rewrite_query(FakeLLM("   "), "What about fees?", HISTORY, 3)

    assert query == "What about fees?"
    assert was_rewritten is False


def test_a_rambling_rewrite_falls_back_to_the_original_question():
    llm = FakeLLM("Certainly! Here is a rewritten query. " + "padding " * 200)

    query, was_rewritten = rewrite_query(llm, "What about fees?", HISTORY, 3)

    assert query == "What about fees?"
    assert was_rewritten is False


def test_evidence_is_numbered_from_one_and_labelled_with_its_source():
    chunks = retrieved(
        make_chunk(0, "Fifty units is the maximum.", page=47, section="Eligibility"),
        make_chunk(1, "International applicants differ.", page=51),
    )

    evidence, used = build_evidence(chunks, max_tokens=1000)

    assert "[1] Test Document | Page 47 | Eligibility" in evidence
    assert "[2] Test Document | Page 51" in evidence
    assert len(used) == 2


def test_evidence_is_trimmed_to_the_token_budget_keeping_the_best_chunks():
    chunks = retrieved(*[make_chunk(i, "word " * 200, page=i + 1) for i in range(5)])

    evidence, used = build_evidence(chunks, max_tokens=300)

    assert 0 < len(used) < 5
    assert estimate_tokens(evidence) <= 300 + estimate_tokens(chunks[0].chunk.text)
    assert used == chunks[: len(used)], "trimming must drop the lowest ranked chunks"


def test_at_least_one_chunk_survives_a_tiny_budget():
    chunks = retrieved(make_chunk(0, "word " * 500))

    _, used = build_evidence(chunks, max_tokens=10)

    assert len(used) == 1


def test_evidence_numbering_matches_the_returned_chunks():
    chunks = retrieved(*[make_chunk(i, "chunk number {}".format(i), page=i + 1) for i in range(4)])

    evidence, used = build_evidence(chunks, max_tokens=1000)

    # Citation [n] must resolve to used[n-1]; the whole citation guarantee
    # depends on these two staying in step.
    for position, item in enumerate(used, start=1):
        assert "[{}]".format(position) in evidence
        assert item.chunk.text in evidence.split("[{}]".format(position))[1]


def test_no_retrieved_chunks_produces_no_evidence():
    evidence, used = build_evidence([], max_tokens=1000)

    assert evidence == ""
    assert used == []
