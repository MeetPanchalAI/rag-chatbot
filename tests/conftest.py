"""Shared fixtures and fakes.

Every test runs offline. The embedder and the LLM are replaced by fakes, and
test PDFs are generated in memory, so there is no network call and no binary
fixture to keep in the repository.
"""

import io
import re
import zlib

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.config import Settings
from app.schemas import Chunk, RetrievedChunk
from app.vector_store import VectorStore

DIM = 64
PAGE_WIDTH, PAGE_HEIGHT = letter
_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, and similar text scores
    higher than unrelated text, which is all the retrieval tests need."""

    def __init__(self, dim: int = DIM, fail_with: Exception | None = None):
        self.dim = dim
        self.fail_with = fail_with
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector


class FakeLLM:
    """Returns scripted replies in order. An Exception in the script is raised."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_pdf(
    pages: list[list],
    title: str = "Test Document",
    toc: list[tuple[str, int]] | None = None,
    columns: int = 1,
) -> bytes:
    """Build a PDF in memory.

    Each page is a list of lines; a line is a string or (text, fontsize, bold).
    `toc` entries are (title, 1-based page number) and become real PDF
    bookmarks. `columns=2` lays each page out in two columns.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(title)
    bookmarks = {page: name for name, page in (toc or [])}

    for page_number, lines in enumerate(pages, start=1):
        column_width = (PAGE_WIDTH - 144) / columns
        per_column = (len(lines) + columns - 1) // columns
        for column in range(columns):
            x = 72 + column * (column_width + 12)
            y = PAGE_HEIGHT - 72
            for line in lines[column * per_column : (column + 1) * per_column]:
                text, size, bold = _line_spec(line)
                pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
                pdf.drawString(x, y, text)
                y -= size * 1.8
        if page_number in bookmarks:
            key = "p{}".format(page_number)
            pdf.bookmarkPage(key)
            pdf.addOutlineEntry(bookmarks[page_number], key, level=0)
        pdf.showPage()

    pdf.save()
    return buffer.getvalue()


def _line_spec(line) -> tuple[str, float, bool]:
    if isinstance(line, tuple):
        text, size = line[0], line[1]
        bold = line[2] if len(line) > 2 else False
        return text, size, bold
    return line, 11, False


def make_chunk(
    index: int,
    text: str,
    doc_id: str = "doc1",
    page: int = 1,
    section: str | None = None,
    title: str = "Test Document",
) -> Chunk:
    return Chunk(
        chunk_id="{}:{:05d}".format(doc_id, index),
        doc_id=doc_id,
        doc_title=title,
        text=text,
        section=section,
        page_start=page,
        page_end=page,
    )


def retrieved(*chunks: Chunk) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunk, score=round(1.0 - i * 0.1, 3))
        for i, chunk in enumerate(chunks)
    ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        data_dir=tmp_path / "data",
        eval_dir=tmp_path / "eval",
        chunk_tokens=120,
        chunk_overlap_tokens=20,
        retriever_top_k=3,
        max_context_tokens=400,
        max_history_turns=3,
    )


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(settings.data_dir)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def simple_pdf() -> bytes:
    """Two pages, a real table of contents, and two clear sections."""
    return make_pdf(
        pages=[
            [
                ("Eligibility Requirements", 18),
                "Applicants must be at least eighteen years old and hold a",
                "valid national identity document issued by their country.",
                "The maximum amount allowed is 50 units per applicant.",
                "Applications are reviewed by the admissions office within",
                "thirty working days of the date of submission.",
            ],
            [
                ("International Applicants", 18),
                "International applicants must also provide a certified",
                "translation of every supporting document they submit.",
                "Processing for international applicants takes sixty days.",
                "The admissions office is responsible for this process.",
            ],
        ],
        title="Handbook",
        toc=[("Eligibility Requirements", 1), ("International Applicants", 2)],
    )
