"""Ingestion: parsing, heading detection, chunking, and the failure paths."""

import io

import pytest
from pypdf import PdfWriter

from app.errors import IngestError
from app.ingestion import (
    Line,
    _assign_columns,
    _split_into_sections,
    build_chunks,
    detect_headings,
    embedding_text,
    ingest_pdf,
    parse_pdf,
)
from app.text_utils import estimate_tokens
from tests.conftest import make_pdf


def test_parses_text_title_and_pages(simple_pdf):
    parsed = parse_pdf(simple_pdf, "handbook.pdf")

    assert parsed.title == "Handbook"
    assert parsed.pages == 2
    assert any("eighteen years old" in line.text for line in parsed.lines)


def test_falls_back_to_filename_when_pdf_has_no_title():
    body = ["Line {} of some ordinary body text on this page.".format(i) for i in range(8)]
    data = make_pdf([body], title="")
    assert parse_pdf(data, "quarterly-report.pdf").title == "quarterly-report"


def test_rejects_a_file_that_is_not_a_pdf():
    with pytest.raises(IngestError):
        parse_pdf(b"this is plainly not a pdf", "broken.pdf")


def test_rejects_a_pdf_with_no_extractable_text():
    # An image-only or blank PDF. We do not run OCR, so this must fail loudly
    # rather than index an empty document.
    with pytest.raises(IngestError, match="no extractable text"):
        parse_pdf(make_pdf([[], []]), "scanned.pdf")


def test_rejects_an_encrypted_pdf(simple_pdf):
    writer = PdfWriter(clone_from=io.BytesIO(simple_pdf))
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(IngestError, match="password protected"):
        parse_pdf(buffer.getvalue(), "locked.pdf")


def test_uses_the_table_of_contents_for_section_titles(simple_pdf, settings):
    parsed = parse_pdf(simple_pdf, "handbook.pdf")
    chunks = build_chunks(parsed, "doc" * 4, settings)

    sections = {chunk.section for chunk in chunks}
    assert "Eligibility Requirements" in sections
    assert "International Applicants" in sections


def test_falls_back_to_page_only_citations_without_headings(settings):
    # Uniform body text, no outline: nothing reliable to call a heading.
    data = make_pdf(
        [
            [
                "The office reviews each application in the order received.",
                "Every application is logged against a reference number.",
                "Decisions are communicated by post within ten working days.",
                "Appeals must be lodged within twenty working days.",
            ]
        ],
        title="Notes",
    )
    parsed = parse_pdf(data, "notes.pdf")

    assert detect_headings(parsed.lines, parsed.toc_titles) == {}
    chunks = build_chunks(parsed, "doc" * 4, settings)
    assert chunks and all(chunk.section is None for chunk in chunks)


def test_reads_two_column_pages_one_column_at_a_time():
    left = ["LEFT sentence number {}".format(i) for i in range(6)]
    right = ["RIGHT sentence number {}".format(i) for i in range(6)]
    parsed = parse_pdf(make_pdf([left + right], columns=2), "two-column.pdf")

    order = [line.text.split()[0] for line in parsed.lines]
    assert order == ["LEFT"] * 6 + ["RIGHT"] * 6, "columns were interleaved"


def test_every_chunk_carries_the_metadata_its_citation_needs(simple_pdf, settings):
    _, chunks = ingest_pdf(simple_pdf, "handbook.pdf", settings)

    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.doc_id and chunk.doc_title
        assert 1 <= chunk.page_start <= chunk.page_end <= 2


def test_chunks_stay_within_the_configured_size(settings):
    body = ["Sentence number {} of a long and uneventful page.".format(i) for i in range(120)]
    parsed = parse_pdf(make_pdf([body[:60], body[60:]]), "long.pdf")
    chunks = build_chunks(parsed, "doc" * 4, settings)

    assert len(chunks) > 1
    # Allow one line of slack: a chunk is closed after the line that crosses it.
    assert all(estimate_tokens(c.text) <= settings.chunk_tokens * 1.5 for c in chunks)


def test_the_same_file_always_gets_the_same_document_id(simple_pdf, settings):
    first, _ = ingest_pdf(simple_pdf, "handbook.pdf", settings)
    second, _ = ingest_pdf(simple_pdf, "renamed.pdf", settings)

    assert first.doc_id == second.doc_id


def test_embedded_text_is_prefixed_with_document_and_section(simple_pdf, settings):
    _, chunks = ingest_pdf(simple_pdf, "handbook.pdf", settings)
    chunk = next(c for c in chunks if c.section)

    text = embedding_text(chunk)
    assert text.startswith(chunk.doc_title)
    assert chunk.section in text
    assert chunk.text in text


# --- regressions found by running the parser over a real-world PDF ---


def _words(count: int, x0: float, x1: float) -> list[dict]:
    return [{"x0": x0, "x1": x1, "top": i * 12.0} for i in range(count)]


def test_a_genuine_two_column_page_is_split():
    words = _words(12, 60, 260) + _words(12, 340, 540)

    assert set(_assign_columns(words, 612)) == {0, 1}


def test_text_crossing_the_middle_vetoes_the_two_column_reading():
    # A centred title or a full-width paragraph spans the gutter. Splitting the
    # page in half would cut it in two and scramble the reading order.
    words = _words(12, 60, 260) + _words(12, 340, 540) + _words(3, 120, 500)

    assert set(_assign_columns(words, 612)) == {0}


def _line(text: str, size: float = 11, bold: bool = False) -> Line:
    return Line(page=1, block=0, text=text, size=size, bold=bold)


def _body(count: int) -> list[Line]:
    """Ordinary body lines, so the median font size reflects a real page."""
    return [_line("Body sentence number {} on this page.".format(i)) for i in range(count)]


def test_the_tail_of_a_wrapped_sentence_is_not_a_heading():
    lines = (
        [_line("Introduction", 18)]
        + _body(6)
        # A sentence that wrapped onto a second line, set in the same large type
        # as the title above it. Only the lowercase start gives it away.
        + [_line("explanation of trade-offs, not a large product", 18)]
        + [_line("Methods", 18)]
        + _body(6)
    )

    headings = detect_headings(lines, toc_titles=set())

    assert set(headings.values()) == {"Introduction", "Methods"}


def test_bullet_points_are_not_headings():
    lines = (
        [_line("Engineering requirements", 18)]
        + [_line("- Validation: validate requests and handle invalid input", 14)]
        + [_line("- Testing: include automated tests for important components", 14)]
        + _body(6)
        + [_line("Evaluation", 18)]
        + _body(6)
    )

    headings = detect_headings(lines, toc_titles=set())

    assert set(headings.values()) == {"Engineering requirements", "Evaluation"}


def test_a_stray_one_line_section_is_folded_into_the_section_above_it():
    # Styled pages produce "headings" with nothing under them. Emitting those as
    # their own chunks fills the index with text too short to retrieve on.
    lines = (
        [_line("Introduction", 18)]
        + [_line("Body sentence number {} of the introduction.".format(i)) for i in range(10)]
        + [_line("Stray Line", 18)]
    )

    sections = _split_into_sections(lines, {0: "Introduction", 11: "Stray Line"})

    assert len(sections) == 1
    assert sections[0][0] == "Introduction"
    assert "Stray Line" in " ".join(line.text for line in sections[0][1]), "text was lost"
