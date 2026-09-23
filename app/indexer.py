"""Ingest a PDF end to end: parse, chunk, embed, store."""

import logging
import time
from typing import Callable

from app import logs
from app.config import Settings
from app.ingestion import embedding_text, ingest_pdf
from app.providers import Embedder
from app.schemas import DocumentInfo, IngestResponse
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

SOURCE_DIR = "documents"

Progress = Callable[[int, int], None]


def index_pdf(
    data: bytes,
    filename: str,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    progress: Progress | None = None,
) -> IngestResponse:
    """Parse, embed and store a PDF. Re-ingesting the same file is a no-op.

    The document id is a hash of the file bytes, so the same PDF uploaded twice
    is recognised rather than indexed again.
    """
    # One id for the whole ingest, so its stages read together in the log.
    with logs.request():
        return _index(data, filename, store, embedder, settings, progress)


def _index(
    data: bytes,
    filename: str,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    progress: Progress | None,
) -> IngestResponse:
    started = time.perf_counter()
    log.debug("parsing %s (%.1f MB)", filename, len(data) / 1_048_576)

    info, chunks = ingest_pdf(data, filename, settings)
    log.debug("parsed %d pages into %d chunks", info.pages, info.chunks)

    if store.has_document(info.doc_id):
        existing = store.get_document(info.doc_id)
        log.info("already indexed: %s, skipping", existing.filename)
        return _response(existing, duplicate=True)

    texts = [embedding_text(chunk) for chunk in chunks]
    vectors: list[list[float]] = []
    batch = max(1, settings.embed_batch_size)

    # Batched because a large PDF is thousands of chunks, and one request per
    # chunk would take hours.
    for start in range(0, len(texts), batch):
        vectors.extend(embedder.embed(texts[start : start + batch]))
        done = min(start + batch, len(texts))
        log.debug("embedded %d/%d chunks", done, len(texts))
        if progress:
            progress(done, len(texts))

    source_file = None
    if settings.keep_source_pdf:
        # Without the original there is no way to re-chunk or re-embed this
        # document later without being handed the file again.
        folder = store.dir / SOURCE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (info.doc_id + ".pdf")).write_bytes(data)
        source_file = "{}/{}.pdf".format(SOURCE_DIR, info.doc_id)
        log.debug("kept the source PDF at %s", source_file)

    store.add(info, chunks, vectors, source_file=source_file)
    log.info(
        "indexed %s | %d pages, %d chunks | %.1fs",
        info.filename, info.pages, info.chunks, time.perf_counter() - started,
    )
    return _response(info, duplicate=False)


def _response(info: DocumentInfo, duplicate: bool) -> IngestResponse:
    return IngestResponse(
        doc_id=info.doc_id,
        filename=info.filename,
        title=info.title,
        pages=info.pages,
        chunks=info.chunks,
        duplicate=duplicate,
    )
