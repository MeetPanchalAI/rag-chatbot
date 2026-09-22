"""Ingest a PDF end to end: parse, chunk, embed, store."""

import logging
from typing import Callable

from app.config import Settings
from app.ingestion import embedding_text, ingest_pdf
from app.providers import Embedder
from app.schemas import DocumentInfo, IngestResponse
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

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
    info, chunks = ingest_pdf(data, filename, settings)

    if store.has_document(info.doc_id):
        existing = store.get_document(info.doc_id)
        log.info("Document %s is already indexed; skipping.", existing.doc_id[:12])
        return _response(existing, duplicate=True)

    texts = [embedding_text(chunk) for chunk in chunks]
    vectors: list[list[float]] = []
    batch = max(1, settings.embed_batch_size)

    # Batched because a large PDF is thousands of chunks, and one request per
    # chunk would take hours.
    for start in range(0, len(texts), batch):
        vectors.extend(embedder.embed(texts[start : start + batch]))
        if progress:
            progress(min(start + batch, len(texts)), len(texts))

    store.add(info, chunks, vectors)
    log.info("Indexed %s: %d pages, %d chunks.", info.filename, info.pages, info.chunks)
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
