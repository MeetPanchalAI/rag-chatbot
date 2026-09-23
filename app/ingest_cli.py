"""Ingest PDFs from the command line.

    python -m app.ingest_cli docs/handbook.pdf [more.pdf ...]

Preferred over the HTTP endpoint for large files: a long document is thousands
of chunks and several minutes of embedding calls, which would outlast a typical
HTTP client timeout.
"""

import argparse
import logging
import sys
from pathlib import Path

from app import logs
from app.config import get_settings
from app.db import Database
from app.errors import AppError
from app.indexer import index_pdf
from app.providers import OpenAIEmbedder
from app.vector_store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the local index.")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files to ingest")
    args = parser.parse_args(argv)

    logs.configure(get_settings().log_level)
    settings = get_settings()
    store = VectorStore(Database(settings.data_dir / "app.db"), settings.data_dir)
    store.load()
    embedder = OpenAIEmbedder(settings)

    failures = 0
    for path in args.paths:
        if not path.is_file():
            print("not found: {}".format(path), file=sys.stderr)
            failures += 1
            continue

        def progress(done: int, total: int, name: str = path.name) -> None:
            print("  {}: embedded {}/{} chunks".format(name, done, total), end="\r")

        try:
            result = index_pdf(
                path.read_bytes(), path.name, store, embedder, settings, progress
            )
        except AppError as exc:
            print("\n{}: {}".format(path.name, exc.message), file=sys.stderr)
            failures += 1
            continue

        status = "already indexed" if result.duplicate else "indexed"
        print(
            "\n{} {}: {} pages, {} chunks, doc_id={}".format(
                status, result.filename, result.pages, result.chunks, result.doc_id
            )
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
