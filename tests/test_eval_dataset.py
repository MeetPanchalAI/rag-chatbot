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


def test_the_set_is_the_size_the_brief_asks_for(rows):
    assert 15 <= len(rows) <= 20


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
        assert row["expected_answer_contains"] == []


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
