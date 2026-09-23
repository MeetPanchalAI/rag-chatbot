"""Logging setup.

Two levels, and the split between them is the whole design:

  INFO  one line per unit of work — a question answered, a PDF indexed, an
        evaluation run. This is what you want running normally.
  DEBUG one line per stage inside it — rewrite, retrieve, rerank, generate.
        This is what you want when a specific answer looks wrong.

Every line carries a short request id, so the stages of one question can be read
together even when several are in flight. The id travels in a context variable
rather than through function arguments, so the modules doing the work stay
unaware of it.

Third-party loggers are quieted: httpx logs a line per HTTP call and pdfminer is
overwhelming at DEBUG, and neither tells you anything about this system.
"""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s [%(request_id)s] %(message)s"
TIME_FORMAT = "%H:%M:%S"

# Libraries that are chatty and say nothing about this system.
QUIET = ("httpx", "httpcore", "openai", "pdfminer", "pypdf", "PIL", "urllib3")


class _AddRequestId(logging.Filter):
    """Put the current request id on every record, including library ones."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def configure(level: str = "INFO") -> None:
    """Install one handler on the root logger. Safe to call more than once."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=TIME_FORMAT))
    handler.addFilter(_AddRequestId())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def request(request_id: str | None = None) -> Iterator[str]:
    """Tag every log line inside this block with one id."""
    token = _request_id.set(request_id or uuid.uuid4().hex[:8])
    try:
        yield _request_id.get()
    finally:
        _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()
