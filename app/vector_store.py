"""Chunk and embedding storage with cosine search.

A local file-backed index: chunks in JSON Lines, embeddings in one numpy array,
document metadata in a small JSON file. No database to install or run.

Embeddings are stored normalised, so cosine similarity is a plain dot product
and search is a single matrix multiply. At a few tens of thousands of chunks
this is well under a millisecond, which is smaller than one network round trip
to the embedding API. A real vector database earns its place when the corpus
outgrows memory or needs shared writers; see DESIGN.md.
"""

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from app.errors import DocumentNotFound
from app.schemas import Chunk, DocumentInfo, RetrievedChunk

log = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
DOCUMENTS_FILE = "documents.json"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.documents: dict[str, DocumentInfo] = {}
        self._lock = threading.Lock()

    # --- persistence ---

    def load(self) -> None:
        """Read the index from disk. An empty or missing index is not an error."""
        chunks_path = self.dir / CHUNKS_FILE
        embeddings_path = self.dir / EMBEDDINGS_FILE
        documents_path = self.dir / DOCUMENTS_FILE

        if not chunks_path.exists() or not embeddings_path.exists():
            log.info("No index found in %s; starting empty.", self.dir)
            return

        with chunks_path.open(encoding="utf-8") as handle:
            self.chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        self.embeddings = np.load(embeddings_path)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise RuntimeError(
                "Index is inconsistent: {} chunks but {} embeddings. "
                "Delete the data directory and re-ingest.".format(
                    len(self.chunks), self.embeddings.shape[0]
                )
            )

        if documents_path.exists():
            raw = json.loads(documents_path.read_text(encoding="utf-8"))
            self.documents = {k: DocumentInfo.model_validate(v) for k, v in raw.items()}

        log.info("Loaded %d chunks from %d documents.", len(self.chunks), len(self.documents))

    def _write(self, new_chunks: list[Chunk]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

        with (self.dir / CHUNKS_FILE).open("a", encoding="utf-8") as handle:
            for chunk in new_chunks:
                handle.write(chunk.model_dump_json() + "\n")

        def write_embeddings(path: Path) -> None:
            # np.save appends ".npy" unless given an open file object.
            with path.open("wb") as handle:
                np.save(handle, self.embeddings, allow_pickle=False)

        _replace(self.dir / EMBEDDINGS_FILE, write_embeddings)
        _replace(
            self.dir / DOCUMENTS_FILE,
            lambda p: p.write_text(
                json.dumps({k: v.model_dump() for k, v in self.documents.items()}, indent=2),
                encoding="utf-8",
            ),
        )

    # --- writing ---

    def add(self, info: DocumentInfo, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Got {} chunks but {} vectors.".format(len(chunks), len(vectors)))
        if not chunks:
            return

        matrix = _normalise(np.asarray(vectors, dtype=np.float32))

        with self._lock:
            if self.embeddings.size and matrix.shape[1] != self.embeddings.shape[1]:
                raise RuntimeError(
                    "Embedding dimension {} does not match the stored index ({}). "
                    "The embedding model changed; re-ingest into a fresh data "
                    "directory.".format(matrix.shape[1], self.embeddings.shape[1])
                )
            self.chunks.extend(chunks)
            self.embeddings = (
                matrix if not self.embeddings.size else np.vstack([self.embeddings, matrix])
            )
            self.documents[info.doc_id] = info
            self._write(chunks)

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

    def search(
        self, query_vector: list[float], top_k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks, optionally within one document."""
        if doc_id is not None and not self.has_document(doc_id):
            raise DocumentNotFound("No document with id " + doc_id)
        if not self.chunks or top_k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        if query.shape[0] != self.embeddings.shape[1]:
            raise RuntimeError("Query embedding does not match the stored index dimension.")

        scores = self.embeddings @ query

        if doc_id is not None:
            mask = np.array([c.doc_id == doc_id for c in self.chunks])
            candidates = np.flatnonzero(mask)
        else:
            candidates = np.arange(len(self.chunks))
        if candidates.size == 0:
            return []

        k = min(top_k, candidates.size)
        best = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
        return [
            RetrievedChunk(chunk=self.chunks[int(i)], score=round(float(scores[i]), 6))
            for i in best
        ]


def _replace(path: Path, write) -> None:
    """Write via a temporary file so a crash cannot truncate the index."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)
