"""Ingest a PDF end to end: parse, chunk, embed, store."""

import logging
from typing import Callable

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

    source_file = None
    if settings.keep_source_pdf:
        # Without the original there is no way to re-chunk or re-embed this
        # document later without being handed the file again.
        folder = store.dir / SOURCE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (info.doc_id + ".pdf")).write_bytes(data)
        source_file = "{}/{}.pdf".format(SOURCE_DIR, info.doc_id)

    store.add(info, chunks, vectors, source_file=source_file)
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
