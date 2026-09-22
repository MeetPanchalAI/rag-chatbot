"""Provider retry policy.

These are the failure paths that only appear against a live API, so they are
driven here with the SDK's own exception types rather than a fake.
"""

import httpx
import openai
import pytest

from app.errors import ProviderError, ProviderTimeout
from app.providers import _retry_once

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def failing(exc: Exception):
    """A call that always raises, counting how many times it was attempted."""
    calls: list[int] = []

    def call():
        calls.append(1)
        raise exc

    return call, calls


def status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        "boom", response=httpx.Response(code, request=REQUEST), body=None
    )


def test_a_successful_call_is_not_retried():
    calls: list[int] = []

    result = _retry_once(lambda: (calls.append(1), "ok")[1], "test call")

    assert result == "ok"
    assert len(calls) == 1


def test_a_timeout_is_retried_once_then_reported():
    call, calls = failing(openai.APITimeoutError(request=REQUEST))

    with pytest.raises(ProviderTimeout):
        _retry_once(call, "test call")

    assert len(calls) == 2


def test_a_rate_limit_is_retried():
    call, calls = failing(
        openai.RateLimitError(
            "slow down", response=httpx.Response(429, request=REQUEST), body=None
        )
    )

    with pytest.raises(ProviderError):
        _retry_once(call, "test call")

    assert len(calls) == 2


def test_a_server_error_is_retried():
    call, calls = failing(status_error(503))

    with pytest.raises(ProviderError):
        _retry_once(call, "test call")

    assert len(calls) == 2


def test_a_rejected_request_is_not_retried():
    # An unsupported parameter fails identically the second time. Retrying only
    # doubles the latency and the cost, and hides the real message.
    call, calls = failing(status_error(400))

    with pytest.raises(ProviderError, match="boom"):
        _retry_once(call, "test call")

    assert len(calls) == 1


def test_a_recovered_call_returns_the_second_result():
    attempts: list[int] = []

    def call():
        attempts.append(1)
        if len(attempts) == 1:
            raise openai.APITimeoutError(request=REQUEST)
        return "recovered"

    assert _retry_once(call, "test call") == "recovered"
    assert len(attempts) == 2
