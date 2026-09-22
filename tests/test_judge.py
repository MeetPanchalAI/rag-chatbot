"""The judge, and what happens when it misbehaves.

An unscored question is honest. A defaulted score would move the averages and
look like a change in the system, which is the one thing an evaluation must
never do.
"""

import json

from app.judge import SCORES, judge_answer
from tests.conftest import FakeLLM

ARGS = dict(
    question="What is the maximum amount?",
    history=[],
    expected="The maximum is 50 units.",
    answer="Fifty units.",
    evidence="[1] Handbook | Page 47\nThe maximum amount allowed is 50 units.",
    citations=['Page 47 - "Eligibility"'],
)


def verdict(**scores) -> str:
    body = {"correctness": 2, "groundedness": 2, "citation_support": 2,
            "completeness": 2, "reason": "matches the reference"}
    body.update(scores)
    return json.dumps(body)


def test_a_well_formed_verdict_is_read_back():
    result = judge_answer(FakeLLM(verdict(groundedness=1)), **ARGS)

    assert result.scores == {"correctness": 2, "groundedness": 1,
                             "citation_support": 2, "completeness": 2}
    assert result.failed is False
    assert result.reason == "matches the reference"


def test_a_fenced_verdict_is_still_read():
    result = judge_answer(FakeLLM("```json\n" + verdict() + "\n```"), **ARGS)

    assert result.scores["correctness"] == 2


def test_the_judge_sees_the_reference_answer_the_evidence_and_the_citations():
    llm = FakeLLM(verdict())

    judge_answer(llm, **ARGS)

    prompt = llm.calls[0]["user"]
    assert ARGS["expected"] in prompt
    assert "The maximum amount allowed is 50 units." in prompt
    assert 'Page 47 - "Eligibility"' in prompt
    assert llm.calls[0]["json_mode"] is True


def test_unparseable_output_scores_nothing_rather_than_guessing():
    result = judge_answer(FakeLLM("I think it was pretty good honestly"), **ARGS)

    assert result.failed is True
    assert all(result.scores[name] is None for name in SCORES)


def test_a_score_outside_the_scale_is_rejected_whole():
    # A judge inventing a 7 has not understood the task; keeping its other
    # numbers would be trusting the same reply we just caught being wrong.
    result = judge_answer(FakeLLM(verdict(correctness=7)), **ARGS)

    assert result.failed is True
    assert all(result.scores[name] is None for name in SCORES)


def test_a_judge_that_errors_does_not_bring_the_run_down():
    result = judge_answer(FakeLLM(RuntimeError("judge model is down")), **ARGS)

    assert result.failed is True
    assert all(result.scores[name] is None for name in SCORES)
    assert "down" in result.reason


def test_an_answer_with_no_citation_is_still_judged():
    result = judge_answer(FakeLLM(verdict(citation_support=0)),
                          **{**ARGS, "citations": []})

    assert result.scores["citation_support"] == 0
