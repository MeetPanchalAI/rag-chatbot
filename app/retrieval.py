"""Query rewriting, dense retrieval, and evidence assembly."""

import logging

from app.providers import Embedder, LLM
from app.schemas import Chunk, Message, RetrievedChunk
from app.text_utils import estimate_tokens
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "You rewrite the user's latest question into a standalone search query.\n"
    "Replace pronouns and implicit references with what they refer to in the "
    "conversation.\n"
    "If the question already stands on its own, repeat it unchanged.\n"
    "Reply with the query only: no preamble, no quotes, no explanation."
)


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

    rewritten = llm.complete(REWRITE_SYSTEM, prompt).strip().strip('"')

    # A rewrite that is empty or rambling is worse than the original question.
    if not rewritten or len(rewritten) > max(400, len(question) * 6):
        log.warning("Discarding an unusable query rewrite; using the original question.")
        return question, False
    return rewritten, rewritten.lower() != question.strip().lower()


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    top_k: int,
    doc_id: str | None = None,
) -> list[RetrievedChunk]:
    vectors = embedder.embed([query])
    if not vectors:
        return []
    return store.search(vectors[0], top_k=top_k, doc_id=doc_id)


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
