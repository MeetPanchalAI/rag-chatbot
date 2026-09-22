"""Embedding and LLM providers.

Everything above this layer talks to the two Protocols, so tests can swap in
fakes and run offline.
"""

import logging
from typing import Protocol

from app.config import Settings
from app.errors import ProviderError, ProviderTimeout

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one call. Callers do the batching."""
        ...


class LLM(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Return the model's text reply."""
        ...


def _retry_once(fn, what: str):
    """Call fn, retrying once on a provider failure. Then give up."""
    import openai

    last: Exception | None = None
    for attempt in (1, 2):
        try:
            return fn()
        except openai.APITimeoutError as exc:
            last = ProviderTimeout(f"{what} timed out")
            log.warning("%s timed out (attempt %d): %s", what, attempt, exc)
        except openai.OpenAIError as exc:
            last = ProviderError(f"{what} failed: {exc}")
            log.warning("%s failed (attempt %d): %s", what, attempt, exc)
    raise last  # type: ignore[misc]


def _client(settings: Settings):
    import openai

    if not settings.openai_api_key:
        raise ProviderError("OPENAI_API_KEY is not set")
    return openai.OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.request_timeout_s,
        max_retries=0,  # we retry ourselves so the behaviour is testable
    )


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        self._client = _client(settings)
        self._model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = _retry_once(
            lambda: self._client.embeddings.create(model=self._model, input=texts),
            "embedding request",
        )
        return [d.embedding for d in resp.data]


class OpenAILLM:
    def __init__(self, settings: Settings, model: str | None = None):
        self._client = _client(settings)
        self._model = model or settings.llm_model
        self._temperature = settings.llm_temperature
        self._reasoning_effort = settings.llm_reasoning_effort.strip()

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Reasoning models reject some parameters. Leaving the env var blank
        # omits it entirely rather than sending an unsupported value.
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = _retry_once(
            lambda: self._client.chat.completions.create(**kwargs), "LLM request"
        )
        return resp.choices[0].message.content or ""
