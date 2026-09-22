"""Generation: the guards between the model and the response."""

import json

from app.generation import REFUSAL, generate_answer
from tests.conftest import FakeLLM

EVIDENCE = "[1] Handbook | Page 47 | Eligibility\nThe maximum amount allowed is 50 units."


def reply(answer: str, answerable: bool = True, citations: list[int] | None = None) -> str:
    return json.dumps(
        {"answer": answer, "answerable": answerable, "citations": citations or [1]}
    )


def test_a_well_formed_answer_passes_through():
    llm = FakeLLM(reply("The maximum amount allowed is 50 units."))

    result = generate_answer(llm, "What is the maximum?", [], EVIDENCE, 1, 3)

    assert result.answerable is True
    assert result.answer == "The maximum amount allowed is 50 units."
    assert result.citations == [1]
    assert llm.calls[0]["json_mode"] is True


def test_json_wrapped_in_a_markdown_fence_is_still_accepted():
    llm = FakeLLM("```json\n" + reply("Fifty units.") + "\n```")

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.answerable is True
    assert result.answer == "Fifty units."


def test_malformed_output_is_retried_once_and_then_succeeds():
    llm = FakeLLM("I'm afraid I cannot do that", reply("Fifty units."))

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.answerable is True
    assert "json_retry" in result.guards
    assert len(llm.calls) == 2


def test_output_that_never_parses_refuses_instead_of_guessing():
    llm = FakeLLM("still not json", "and still not json")

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.answerable is False
    assert result.answer == REFUSAL
    assert result.citations == []
    assert "json_failed" in result.guards


def test_a_citation_outside_the_evidence_is_dropped():
    llm = FakeLLM(reply("Fifty units.", citations=[1, 7]))

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.citations == [1]
    assert "dropped_citation:7" in result.guards


def test_duplicate_citations_are_collapsed():
    llm = FakeLLM(reply("Fifty units.", citations=[1, 1]))

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.citations == [1]


def test_an_answer_with_no_valid_citation_is_not_presented_as_grounded():
    # The model claims the evidence supports this but points at nothing real.
    llm = FakeLLM(reply("The maximum is 900 units.", citations=[4]))

    result = generate_answer(llm, "How much?", [], EVIDENCE, 1, 3)

    assert result.answerable is False
    assert result.answer == REFUSAL
    assert "answerable_without_citations" in result.guards


def test_the_model_may_say_the_document_does_not_answer_the_question():
    llm = FakeLLM(reply("The document does not mention parking.", False, []))

    result = generate_answer(llm, "Where do I park?", [], EVIDENCE, 1, 3)

    assert result.answerable is False
    assert "parking" in result.answer


def test_empty_retrieval_refuses_without_calling_the_model():
    llm = FakeLLM()

    result = generate_answer(llm, "Where do I park?", [], "", 0, 3)

    assert result.answerable is False
    assert result.answer == REFUSAL
    assert llm.calls == [], "no evidence means there is nothing to ground an answer in"
