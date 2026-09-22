"""Query rewriting, retrieval, and evidence assembly."""

import logging
from collections import defaultdict

from app import prompts
from app.providers import Embedder, LLM
from app.schemas import Chunk, Message, RetrievedChunk
from app.text_utils import estimate_tokens
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

def page_label(chunk: Chunk) -> str:
    if chunk.page_start == chunk.page_end:
        return "Page {}".format(chunk.page_start)
    return "Pages {}-{}".format(chunk.page_start, chunk.page_end)


def rewrite_query(
    llm: LLM, question: str, history: list[Message], max_turns: int
) -> tuple[str, bool]:
    """Turn a follow-up into a standalone query. Returns (query, was_rewritten).

    A question like "what about international applicants?" retrieves nothing on
    its own, so follow-up support needs this step before the search.

    History is used only to interpret the question. It never becomes evidence.
    """
    if not history:
        return question, False

    recent = history[-(max_turns * 2) :]
    transcript = "\n".join("{}: {}".format(m.role, m.content) for m in recent)
    prompt = "Conversation:\n{}\n\nLatest question: {}".format(transcript, question)

    rewritten = llm.complete(prompts.load("rewrite_query"), prompt).strip().strip('"')

    # A rewrite that is empty or rambling is worse than the original question.
    if not rewritten or len(rewritten) > max(400, len(question) * 6):
        log.warning("Discarding an unusable query rewrite; using the original question.")
        return question, False
    return rewritten, rewritten.lower() != question.strip().lower()


def fuse(
    rankings: list[list[RetrievedChunk]], rrf_k: int, top_k: int
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion.

    Each list contributes 1/(rrf_k + rank) to a chunk's score. Fusing on *rank*
    rather than score is the point: a cosine similarity and a BM25 score are not
    on the same scale and cannot be added, but their orderings can be combined.
    """
    totals: dict[str, float] = defaultdict(float)
    items: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            totals[item.chunk.chunk_id] += 1.0 / (rrf_k + rank)
            items.setdefault(item.chunk.chunk_id, item)

    best = sorted(totals, key=lambda chunk_id: -totals[chunk_id])[:top_k]
    return [
        RetrievedChunk(chunk=items[chunk_id].chunk, score=round(totals[chunk_id], 6))
        for chunk_id in best
    ]


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    top_k: int,
    doc_id: str | None = None,
    hybrid: bool = False,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Dense search, optionally fused with BM25 keyword search.

    Dense search handles paraphrase; keyword search catches the exact terms
    embeddings blur together. Neither is reliably better, so hybrid takes both.
    """
    vectors = embedder.embed([query])
    if not vectors:
        return []

    if not hybrid:
        return store.search(vectors[0], top_k=top_k, doc_id=doc_id)

    # Each ranker offers more than we need, so fusion has something to work with.
    pool = max(top_k * 3, 30)
    dense = store.search(vectors[0], top_k=pool, doc_id=doc_id)
    keyword = store.keyword_search(query, top_k=pool, doc_id=doc_id)
    if not keyword:
        return dense[:top_k]
    return fuse([dense, keyword], rrf_k=rrf_k, top_k=top_k)


def build_evidence(
    retrieved: list[RetrievedChunk], max_tokens: int
) -> tuple[str, list[RetrievedChunk]]:
    """Format retrieved chunks as numbered evidence within a token budget.

    Returns the text shown to the model and the chunks it actually contains, in
    the same order. The model cites positions in this list, so the two must stay
    aligned: citation [2] always means the second item returned here.
    """
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0

    for item in retrieved:
        chunk = item.chunk
        header = "[{}] {} | {}".format(len(used) + 1, chunk.doc_title, page_label(chunk))
        if chunk.section:
            header += " | " + chunk.section
        block = header + "\n" + chunk.text
        tokens = estimate_tokens(block)
        if used and total + tokens > max_tokens:
            break
        blocks.append(block)
        used.append(item)
        total += tokens

    return "\n\n".join(blocks), used
