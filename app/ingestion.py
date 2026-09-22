"""PDF parsing and chunking.

The pipeline is deliberately generic: nothing here assumes a particular kind of
document. Every step degrades to a simpler behaviour when the PDF does not
support it, so a badly structured file still produces usable chunks.

    bytes -> lines (column aware) -> headings -> sections -> chunks

Parsing is split from chunking on purpose. Everything below the parsing section
works on a list of Line objects and does not care which library produced them.
"""

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pdfplumber
from pypdf import PdfReader

from app.config import Settings
from app.errors import IngestError
from app.schemas import Chunk, DocumentInfo
from app.text_utils import estimate_tokens, normalize, split_sentences

log = logging.getLogger(__name__)

MIN_USABLE_CHARS = 200
MAX_HEADING_CHARS = 120
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_FRACTION = 0.30  # above this, heading detection is clearly noise
LINE_TOLERANCE = 3.0  # points; words this close vertically are one line
GUTTER_RATIO = 0.015  # half-width of the empty strip a two-column page needs
MIN_SECTION_TOKENS = 15  # below this, a "section" is a stray line, not a section
PARAGRAPH_GAP_RATIO = 1.8  # gap between lines that starts a new paragraph

_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+){0,3}[.)]?\s+\S")
_PAGE_NUMBER_ONLY = re.compile(r"^[ivxlcdm\d\s.-]+$", re.IGNORECASE)
_BULLET = re.compile(r"^[-*•·–—]\s")


@dataclass
class Line:
    """One line of text with the metadata needed to place and classify it."""

    page: int  # 1-based PDF page index
    block: int  # paragraph ordinal within the document, for joining lines
    text: str
    size: float
    bold: bool


@dataclass
class ParsedPDF:
    title: str
    pages: int
    lines: list[Line]
    toc_titles: set[str]


# --- Parsing -----------------------------------------------------------------


def _assign_columns(words: list[dict], page_width: float) -> list[int]:
    """Label each word with its column index.

    Grouping words into lines purely by vertical position interleaves the
    columns of a two-column page into unreadable text. When the layout looks
    like two columns we split on the page midline first.
    """
    mid = page_width / 2
    tol = page_width * 0.02
    left = sum(1 for w in words if w["x1"] <= mid + tol)
    right = sum(1 for w in words if w["x0"] >= mid - tol)
    threshold = max(10, 0.25 * len(words))

    # A real two-column page has an empty strip down the middle. Centred titles
    # and full-width paragraphs cross it, and splitting those in half produces
    # nonsense, so any meaningful crossing vetoes the two-column reading.
    gutter = page_width * GUTTER_RATIO
    crossing = sum(1 for w in words if w["x0"] < mid + gutter and w["x1"] > mid - gutter)

    if left < threshold or right < threshold or crossing > max(2, 0.02 * len(words)):
        return [0] * len(words)
    # Words straddling the midline are full-width headers; keep them in the
    # left column so they stay ahead of the text they introduce.
    return [1 if w["x0"] >= mid - tol else 0 for w in words]


def _page_lines(page, page_number: int) -> list[tuple[int, float, Line]]:
    """Extract one page as (column, vertical position, Line) in reading order."""
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
    except Exception as exc:  # a single damaged page should not kill the ingest
        log.warning("Skipping page %d: %s", page_number, exc)
        return []
    if not words:
        return []

    columns = _assign_columns(words, page.width or 1.0)
    ordered = sorted(
        zip(columns, words), key=lambda pair: (pair[0], pair[1]["top"], pair[1]["x0"])
    )

    lines: list[tuple[int, float, Line]] = []
    current: list[dict] = []
    current_column = ordered[0][0]
    current_top = float(ordered[0][1]["top"])

    def flush() -> None:
        if not current:
            return
        text = " ".join(w["text"] for w in sorted(current, key=lambda w: w["x0"])).strip()
        if not text:
            return
        lines.append(
            (
                current_column,
                current_top,
                Line(
                    page=page_number,
                    block=0,  # assigned once the page is fully read
                    text=text,
                    size=max(float(w.get("size") or 0.0) for w in current),
                    bold=any("bold" in str(w.get("fontname", "")).lower() for w in current),
                ),
            )
        )

    for column, word in ordered:
        if column != current_column or abs(float(word["top"]) - current_top) > LINE_TOLERANCE:
            flush()
            current = []
            current_column = column
            current_top = float(word["top"])
        current.append(word)
    flush()
    return lines


def _assign_blocks(page_lines: list[tuple[int, float, Line]], first_block: int) -> int:
    """Number paragraphs so lines of one paragraph can be joined back together.

    A new paragraph starts at a column change or an unusually large vertical
    gap, measured against the typical line spacing on the page.
    """
    if not page_lines:
        return first_block

    gaps = [b - a for (_, a, _), (_, b, _) in zip(page_lines, page_lines[1:]) if b > a]
    typical = sorted(gaps)[len(gaps) // 2] if gaps else 0.0

    block = first_block
    for index, (column, top, line) in enumerate(page_lines):
        if index:
            previous_column, previous_top, _ = page_lines[index - 1]
            if column != previous_column or (
                typical and top - previous_top > typical * PARAGRAPH_GAP_RATIO
            ):
                block += 1
        line.block = block
    return block + 1


def _read_outline(data: bytes) -> set[str]:
    """Section titles from the table of contents the PDF ships with, if any."""
    try:
        reader = PdfReader(io.BytesIO(data))
        outline = reader.outline
    except Exception as exc:
        log.info("No usable table of contents: %s", exc)
        return set()

    titles: set[str] = set()

    def walk(items) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
            else:
                title = getattr(item, "title", None)
                if title and title.strip():
                    titles.add(normalize(title))

    walk(outline or [])
    return titles


def parse_pdf(data: bytes, filename: str) -> ParsedPDF:
    """Read a PDF into ordered lines. Raises IngestError on anything unusable."""
    try:
        if PdfReader(io.BytesIO(data)).is_encrypted:
            raise IngestError("The PDF is password protected.")
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("Could not open the PDF: " + str(exc)) from exc

    lines: list[Line] = []
    block_offset = 0
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise IngestError("The PDF has no pages.")
            title = str((pdf.metadata or {}).get("Title") or "").strip()

            for page_number, page in enumerate(pdf.pages, start=1):
                page_lines = _page_lines(page, page_number)
                if not page_lines:
                    continue
                block_offset = _assign_blocks(page_lines, block_offset)
                lines.extend(line for _, _, line in page_lines)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("Could not read the PDF: " + str(exc)) from exc

    if sum(len(line.text) for line in lines) < MIN_USABLE_CHARS:
        raise IngestError(
            "The PDF contains no extractable text. It is probably scanned; this "
            "pipeline does not run OCR."
        )

    return ParsedPDF(
        title=title or filename.rsplit(".", 1)[0],
        pages=page_count,
        lines=lines,
        toc_titles=_read_outline(data),
    )


# --- Headings ----------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def detect_headings(lines: list[Line], toc_titles: set[str]) -> dict[int, str]:
    """Return {line index: heading text} for lines that start a new section.

    Three signals, strongest first:
      1. the line matches a title in the table of contents the PDF ships with,
      2. the line is numbered, like "3.2 Eligibility",
      3. the line is larger or bolder than body text and reads like a title.

    If the result looks like noise, or finds almost nothing, we return no
    headings at all and citations fall back to page numbers only.
    """
    if not lines:
        return {}

    body_size = _median([line.size for line in lines])
    headings: dict[int, str] = {}

    for index, line in enumerate(lines):
        text = line.text.strip()
        if len(text) > MAX_HEADING_CHARS or _PAGE_NUMBER_ONLY.match(text):
            continue

        if normalize(text) in toc_titles:
            headings[index] = text
            continue

        if text.endswith((".", ",", ";", ":")) or _BULLET.match(text):
            continue
        # Headings begin like titles. This rejects the wrapped second half of a
        # body sentence, which otherwise looks exactly like a short heading.
        if not (text[0].isupper() or text[0].isdigit()):
            continue
        if _NUMBERED_HEADING.match(text):
            headings[index] = text
        elif body_size and line.size > body_size * HEADING_SIZE_RATIO:
            headings[index] = text
        elif line.bold and len(text.split()) <= 12:
            headings[index] = text

    if len(headings) < 2:
        log.info("No reliable headings found; citations will use page numbers only.")
        return {}
    if len(headings) > len(lines) * MAX_HEADING_FRACTION:
        log.info("Heading detection looked like noise (%d hits); ignoring.", len(headings))
        return {}
    return headings


# --- Chunking ----------------------------------------------------------------


def _join_lines(lines: list[Line]) -> str:
    """Join lines back into paragraphs, undoing end-of-line hyphenation."""
    parts: list[str] = []
    previous_block: int | None = None
    for line in lines:
        text = line.text
        if previous_block is None:
            parts.append(text)
        elif line.block != previous_block:
            parts.append("\n\n" + text)
        elif parts and parts[-1].endswith("-") and text[:1].islower():
            parts[-1] = parts[-1][:-1] + text
        else:
            parts.append(" " + text)
        previous_block = line.block
    return "".join(parts).strip()


def _explode_long_lines(lines: list[Line], max_tokens: int) -> list[Line]:
    """Split any single line larger than a whole chunk into smaller pieces."""
    out: list[Line] = []
    for line in lines:
        if estimate_tokens(line.text) <= max_tokens:
            out.append(line)
            continue
        for piece in split_sentences(line.text) or [line.text]:
            while estimate_tokens(piece) > max_tokens:
                words = piece.split()
                cut = max(1, len(words) * max_tokens // max(1, estimate_tokens(piece)))
                out.append(
                    Line(line.page, line.block, " ".join(words[:cut]), line.size, line.bold)
                )
                piece = " ".join(words[cut:])
            if piece.strip():
                out.append(Line(line.page, line.block, piece, line.size, line.bold))
    return out


def _overlap_tail(lines: list[Line], overlap_tokens: int) -> list[Line]:
    """The trailing lines to repeat at the start of the next chunk."""
    tail: list[Line] = []
    total = 0
    for line in reversed(lines):
        tokens = estimate_tokens(line.text)
        if total + tokens > overlap_tokens:
            break
        tail.insert(0, line)
        total += tokens
    return tail


def _split_into_sections(
    lines: list[Line], headings: dict[int, str]
) -> list[tuple[str | None, list[Line]]]:
    """Group lines under their heading, then absorb the fragments.

    Heading detection is deliberately generous, so a styled page can produce
    "sections" only a line long. Those are not sections; emitting them as their
    own chunks would fill the index with text too short to retrieve on. Each one
    is folded back into the section above it, heading line included so no text
    is lost.
    """
    raw: list[tuple[str | None, Line | None, list[Line]]] = []
    title: str | None = None
    heading_line: Line | None = None
    body: list[Line] = []

    for index, line in enumerate(lines):
        if index in headings:
            if body or heading_line is not None:
                raw.append((title, heading_line, body))
            title = headings[index]
            heading_line = line
            body = []
        else:
            body.append(line)
    if body or heading_line is not None:
        raw.append((title, heading_line, body))

    sections: list[list] = []
    for section_title, line_of_heading, section_body in raw:
        tokens = sum(estimate_tokens(item.text) for item in section_body)
        if sections and tokens < MIN_SECTION_TOKENS:
            if line_of_heading is not None:
                sections[-1][1].append(line_of_heading)
            sections[-1][1].extend(section_body)
        else:
            sections.append([section_title, list(section_body)])

    return [(section_title, body_lines) for section_title, body_lines in sections]


def build_chunks(parsed: ParsedPDF, doc_id: str, settings: Settings) -> list[Chunk]:
    """Split the document into overlapping chunks that respect section bounds."""
    headings = detect_headings(parsed.lines, parsed.toc_titles)
    chunks: list[Chunk] = []

    def emit(section: str | None, lines: list[Line]) -> None:
        if not lines:
            return
        text = _join_lines(lines)
        if not text:
            return
        chunks.append(
            Chunk(
                chunk_id="{}:{:05d}".format(doc_id[:12], len(chunks)),
                doc_id=doc_id,
                doc_title=parsed.title,
                text=text,
                section=section,
                page_start=min(line.page for line in lines),
                page_end=max(line.page for line in lines),
            )
        )

    for section, section_lines in _split_into_sections(parsed.lines, headings):
        buffer: list[Line] = []
        tokens = 0
        for line in _explode_long_lines(section_lines, settings.chunk_tokens):
            line_tokens = estimate_tokens(line.text)
            if buffer and tokens + line_tokens > settings.chunk_tokens:
                emit(section, buffer)
                buffer = _overlap_tail(buffer, settings.chunk_overlap_tokens)
                tokens = sum(estimate_tokens(item.text) for item in buffer)
            buffer.append(line)
            tokens += line_tokens
        emit(section, buffer)

    return chunks


# --- Entry point -------------------------------------------------------------


def embedding_text(chunk: Chunk) -> str:
    """The text we embed, which is not the text we display.

    Chunks from the middle of a section often read as "this gives us the result
    above", which embeds to nothing useful. Prefixing the document and section
    title gives every chunk a topical anchor.
    """
    header = chunk.doc_title
    if chunk.section:
        header = header + " | " + chunk.section
    return header + "\n\n" + chunk.text


def ingest_pdf(
    data: bytes, filename: str, settings: Settings
) -> tuple[DocumentInfo, list[Chunk]]:
    """Parse and chunk a PDF. Does not embed or store anything."""
    doc_id = hashlib.sha256(data).hexdigest()
    parsed = parse_pdf(data, filename)
    chunks = build_chunks(parsed, doc_id, settings)
    if not chunks:
        raise IngestError("The PDF produced no usable chunks.")

    info = DocumentInfo(
        doc_id=doc_id,
        filename=filename,
        title=parsed.title,
        pages=parsed.pages,
        chunks=len(chunks),
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return info, chunks
