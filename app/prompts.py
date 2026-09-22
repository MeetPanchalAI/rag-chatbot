"""Prompts, loaded from the `prompts/` folder.

Instructions to a model are content, not code: they get rewritten far more often
than the functions around them, and a diff on a text file says what changed
without the noise of Python quoting.

Files are read when used, not at import, so editing one takes effect on the next
question without restarting the server.

Substitution is `{name}` replaced literally, never `str.format`, because several
of these prompts contain JSON examples with braces in them.
"""

from pathlib import Path

from app.errors import AppError

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Every prompt the application sends. The test suite checks each one exists.
NAMES = ("answer", "answer_retry", "rewrite_query", "rerank", "judge")


class PromptMissing(AppError):
    status_code = 500
    code = "prompt_missing"


def load(name: str, **values: str) -> str:
    """Read a prompt, optionally substituting {placeholders}."""
    path = PROMPTS_DIR / (name + ".txt")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptMissing("Could not read the prompt {}: {}".format(path, exc)) from exc
    if not text:
        raise PromptMissing("The prompt {} is empty.".format(path))

    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text
