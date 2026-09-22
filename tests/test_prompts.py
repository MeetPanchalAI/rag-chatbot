"""The contracts a prompt edit must not break.

Prompts are content, meant to be rewritten. These check the few things the
surrounding code actually depends on, so curation cannot silently break parsing,
citations or refusal.
"""

import pytest

from app import prompts
from app.prompts import NAMES, PromptMissing

JSON_MODE = ("answer", "answer_retry", "rerank", "judge")


@pytest.mark.parametrize("name", NAMES)
def test_every_prompt_exists_and_says_something(name):
    assert len(prompts.load(name)) > 40


@pytest.mark.parametrize("name", JSON_MODE)
def test_prompts_sent_in_json_mode_contain_the_word_json(name):
    # The API requires "json" to appear in the messages when JSON mode is on.
    assert "json" in prompts.load(name).lower(), (
        "{} is sent with json_mode=True and must mention JSON".format(name)
    )


def test_the_answer_prompt_keeps_the_citation_contract():
    """Citations are evidence numbers we resolve ourselves. If the prompt stops
    asking for numbers, an invented page number could reach the user."""
    text = prompts.load("answer").lower()

    assert "evidence numbers" in text
    assert "citations" in text


def test_the_answer_prompt_can_still_refuse():
    text = prompts.load("answer").lower()

    assert "answerable" in text, "refusal is expressed by setting answerable to false"


def test_the_answer_prompt_forbids_outside_knowledge():
    assert "outside knowledge" in prompts.load("answer").lower()


def test_the_reply_shapes_match_what_the_code_parses():
    answer = prompts.load("answer")
    for key in ("answer", "answerable", "citations"):
        assert key in answer

    assert "order" in prompts.load("rerank")

    judge = prompts.load("judge")
    for key in ("correctness", "groundedness", "citation_support", "completeness"):
        assert key in judge


def test_the_judge_keeps_the_zero_to_two_scale():
    # A score outside 0..2 voids the whole verdict and leaves it unscored.
    assert "0, 1 or 2" in prompts.load("judge")


def test_the_retry_prompt_takes_the_parse_error():
    filled = prompts.load("answer_retry", error="expecting value")

    assert "expecting value" in filled
    assert "{error}" not in filled


def test_substitution_survives_the_json_braces_in_a_prompt():
    # These prompts contain JSON examples, so str.format would raise on them.
    assert "{" in prompts.load("answer"), "the answer prompt shows a JSON shape"
    prompts.load("answer_retry", error="a {brace} in the message")


def test_a_missing_prompt_fails_with_a_clear_message():
    with pytest.raises(PromptMissing, match="not-a-real-prompt"):
        prompts.load("not-a-real-prompt")


def test_editing_a_prompt_takes_effect_without_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    (tmp_path / "answer.txt").write_text("first version of the prompt, long enough", encoding="utf-8")
    assert prompts.load("answer").startswith("first")

    (tmp_path / "answer.txt").write_text("second version of the prompt, long enough", encoding="utf-8")
    assert prompts.load("answer").startswith("second"), "prompts must not be cached"


def test_no_prompt_text_is_left_behind_in_the_code():
    """Instructions live in prompts/. A stray one in code would be edited in the
    wrong place and silently ignored."""
    from pathlib import Path

    for module in Path("app").glob("*.py"):
        source = module.read_text(encoding="utf-8")
        assert "Reply with JSON only, in this shape" not in source, module
        assert "You answer questions about a document" not in source, module
        assert "You grade one answer" not in source, module
