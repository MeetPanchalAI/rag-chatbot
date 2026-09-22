"""Ingestion: parsing, heading detection, chunking, and the failure paths."""

import io

import pytest
from pypdf import PdfWriter

from app.errors import IngestError
from app.ingestion import build_chunks, detect_headings, embedding_text, ingest_pdf, parse_pdf
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
