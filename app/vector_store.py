"""Chunk and vector storage, backed by SQLite.

Chunks and their embeddings live in one table, written in one transaction. At
load the vectors are stacked into a single numpy matrix held in memory, so
search stays what it was: one matrix multiply, well under a millisecond at tens
of thousands of chunks, and far smaller than the embedding call that precedes
it. SQLite is the durable copy; the matrix is a derived cache of it.

A hosted vector database earns its place when the corpus outgrows memory or
needs concurrent writers. Neither is true here, and this way a document can be
deleted with one statement. See DESIGN.md.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.db import Database
from app.errors import DocumentNotFound
from app.schemas import Chunk, DocumentInfo, RetrievedChunk

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Words for the keyword index. No stemming: the gain is small and it hides
    the exact identifiers keyword search exists to catch."""
    return _WORD.findall(text.lower())


LEGACY_CHUNKS = "chunks.jsonl"
LEGACY_EMBEDDINGS = "embeddings.npy"
LEGACY_DOCUMENTS = "documents.json"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, db: Database, data_dir: Path | None = None):
        self.db = db
        self.dir = Path(data_dir) if data_dir else db.path.parent
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.documents: dict[str, DocumentInfo] = {}
        self._bm25: BM25Okapi | None = None

    # --- loading ---

    def load(self) -> None:
        """Build the in-memory index from the database."""
        self.db.setup()
        self._migrate_legacy_files()

        self.documents = {
            row["doc_id"]: _document(row) for row in self.db.query("SELECT * FROM documents")
        }
        rows = self.db.query(
            "SELECT chunk_id, doc_id, text, section, page_start, page_end, embedding "
            "FROM chunks ORDER BY rowid"
        )
        self.chunks = [_chunk(row, self.documents) for row in rows]
        self.embeddings = (
            np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._build_keyword_index()
        log.info(
            "Loaded %d chunks from %d documents.", len(self.chunks), len(self.documents)
        )

    def _migrate_legacy_files(self) -> None:
        """Carry an index written as JSONL + npy into the database.

        Re-embedding would mean having the original PDFs, which the file layout
        never kept, so this imports rather than rebuilds.
        """
        chunks_path = self.dir / LEGACY_CHUNKS
        embeddings_path = self.dir / LEGACY_EMBEDDINGS
        if not chunks_path.exists() or not embeddings_path.exists():
            return
        if self.db.one("SELECT 1 FROM documents LIMIT 1"):
            return

        with chunks_path.open(encoding="utf-8") as handle:
            chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        vectors = np.load(embeddings_path)
        if len(chunks) != vectors.shape[0]:
            raise RuntimeError(
                "Cannot migrate: {} chunks but {} embeddings.".format(
                    len(chunks), vectors.shape[0]
                )
            )

        documents_path = self.dir / LEGACY_DOCUMENTS
        raw = json.loads(documents_path.read_text(encoding="utf-8")) if documents_path.exists() else {}
        infos = [DocumentInfo.model_validate(value) for value in raw.values()]

        log.info("Migrating %d chunks from the file index into SQLite.", len(chunks))
        with self.db.write() as connection:
            for info in infos:
                _insert_document(connection, info, source_file=None)
            per_document: dict[str, int] = {}
            for chunk, vector in zip(chunks, vectors):
                ordinal = per_document.get(chunk.doc_id, 0)
                per_document[chunk.doc_id] = ordinal + 1
                _insert_chunk(connection, chunk, ordinal, vector)

        for name in (LEGACY_CHUNKS, LEGACY_EMBEDDINGS, LEGACY_DOCUMENTS):
            path = self.dir / name
            if path.exists():
                path.rename(path.with_suffix(path.suffix + ".migrated"))
        log.info("Migration complete; the old files were renamed to *.migrated.")

    # --- writing ---

    def add(
        self,
        info: DocumentInfo,
        chunks: list[Chunk],
        vectors: list[list[float]],
        source_file: str | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Got {} chunks but {} vectors.".format(len(chunks), len(vectors)))
        if not chunks:
            return

        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        if self.embeddings.size and matrix.shape[1] != self.embeddings.shape[1]:
            raise RuntimeError(
                "Embedding dimension {} does not match the stored index ({}). The "
                "embedding model changed; re-ingest into a fresh data directory.".format(
                    matrix.shape[1], self.embeddings.shape[1]
                )
            )

        with self.db.write() as connection:
            _insert_document(connection, info, source_file)
            for ordinal, (chunk, vector) in enumerate(zip(chunks, matrix)):
                _insert_chunk(connection, chunk, ordinal, vector)

        self.documents[info.doc_id] = info
        self.chunks.extend(chunks)
        self.embeddings = matrix if not self.embeddings.size else np.vstack([self.embeddings, matrix])
        self._build_keyword_index()

    def delete_document(self, doc_id: str) -> DocumentInfo:
        """Remove a document, its chunks and its stored PDF."""
        info = self.get_document(doc_id)
        # Read the path first: after the row is gone there is nothing to look up.
        source = self.source_path(doc_id)

        with self.db.write() as connection:
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        if source and source.exists():
            source.unlink()

        keep = [i for i, chunk in enumerate(self.chunks) if chunk.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = (
            self.embeddings[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        )
        self.documents.pop(doc_id, None)
        self._build_keyword_index()
        log.info("Deleted document %s (%s).", doc_id[:12], info.filename)
        return info

    def _build_keyword_index(self) -> None:
        """Rebuild the BM25 index. It is derived from the chunks, like the matrix."""
        self._bm25 = BM25Okapi([tokenise(c.text) for c in self.chunks]) if self.chunks else None

    # --- reading ---

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def get_document(self, doc_id: str) -> DocumentInfo:
        if doc_id not in self.documents:
            raise DocumentNotFound("No document with id " + doc_id)
        return self.documents[doc_id]

    def list_documents(self) -> list[DocumentInfo]:
        return list(self.documents.values())

    def source_path(self, doc_id: str) -> Path | None:
        """Where the original PDF was kept, if it was."""
        row = self.db.one("SELECT source_file FROM documents WHERE doc_id = ?", doc_id)
        if not row or not row["source_file"]:
            return None
        return self.dir / row["source_file"]

    def _candidates(self, doc_id: str | None) -> np.ndarray:
        if doc_id is not None and not self.has_document(doc_id):
            raise DocumentNotFound("No document with id " + doc_id)
        if doc_id is None:
            return np.arange(len(self.chunks))
        return np.flatnonzero(np.array([c.doc_id == doc_id for c in self.chunks]))

    def _ranked(self, scores: np.ndarray, candidates: np.ndarray, top_k: int) -> list[int]:
        if candidates.size == 0 or top_k <= 0:
            return []
        k = min(top_k, candidates.size)
        return [int(i) for i in candidates[np.argsort(-scores[candidates], kind="stable")[:k]]]

    def search(
        self, query_vector: list[float], top_k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Dense search: the top_k most similar chunks, optionally within one document."""
        if not self.chunks or top_k <= 0:
            self._candidates(doc_id)  # still raise for an unknown document
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        if query.shape[0] != self.embeddings.shape[1]:
            raise RuntimeError("Query embedding does not match the stored index dimension.")

        scores = self.embeddings @ query
        return [
            RetrievedChunk(chunk=self.chunks[i], score=round(float(scores[i]), 6))
            for i in self._ranked(scores, self._candidates(doc_id), top_k)
        ]

    def keyword_search(
        self, query: str, top_k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        """BM25 over the same chunks. Catches exact terms that embeddings blur."""
        candidates = self._candidates(doc_id)
        # Single characters are indexed but never searched on: in a maths text
        # "k" and "n" appear on nearly every page, so a query containing one
        # matches everything and ranks noise.
        tokens = [t for t in tokenise(query) if len(t) > 1]
        if self._bm25 is None or not tokens or top_k <= 0:
            return []

        # A chunk is a match if it contains a query term. Filtering on the score
        # instead would drop real matches: when a term appears in most of a small
        # corpus, BM25 gives it a negative or floored weight.
        wanted = set(tokens)
        matched = np.array(
            [i for i in candidates if wanted & self._bm25.doc_freqs[i].keys()], dtype=int
        )
        if matched.size == 0:
            return []

        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=round(float(scores[i]), 6))
            for i in self._ranked(scores, matched, top_k)
        ]


# --- row mapping ---


def _insert_document(connection, info: DocumentInfo, source_file: str | None) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO documents "
        "(doc_id, filename, title, pages, chunks, ingested_at, source_file) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (info.doc_id, info.filename, info.title, info.pages, info.chunks,
         info.ingested_at, source_file),
    )


def _insert_chunk(connection, chunk: Chunk, ordinal: int, vector) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO chunks "
        "(chunk_id, doc_id, ordinal, text, section, page_start, page_end, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (chunk.chunk_id, chunk.doc_id, ordinal, chunk.text, chunk.section,
         chunk.page_start, chunk.page_end,
         np.asarray(vector, dtype=np.float32).tobytes()),
    )


def _document(row) -> DocumentInfo:
    return DocumentInfo(
        doc_id=row["doc_id"], filename=row["filename"], title=row["title"],
        pages=row["pages"], chunks=row["chunks"], ingested_at=row["ingested_at"],
    )


def _chunk(row, documents: dict[str, DocumentInfo]) -> Chunk:
    info = documents.get(row["doc_id"])
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_title=info.title if info else row["doc_id"][:12],
        text=row["text"],
        section=row["section"],
        page_start=row["page_start"],
        page_end=row["page_end"],
    )
