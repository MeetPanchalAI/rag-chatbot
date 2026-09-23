"""The chat flow: rewrite, retrieve, generate, cite.

Kept out of the API layer so the whole flow can be tested without HTTP, and so
the evaluation script runs the exact same code path as a real request.
"""

import logging
import time

from app import logs
from app.config import Settings
from app.generation import generate_answer
from app.providers import Embedder, LLM
from app.rerank import rerank
from app.retrieval import build_evidence, page_label, retrieve, rewrite_query
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    RetrievalTraceItem,
    RetrievedChunk,
    Trace,
)
from app.text_utils import estimate_tokens
from app.vector_store import VectorStore

log = logging.getLogger(__name__)


def build_citation(chunk: Chunk, include_document: bool) -> Citation:
    """Build a citation from chunk metadata only.

    The model never writes page numbers or section names. It returns positions
    in the evidence list, and we resolve those positions here against the
    chunks we retrieved. A page number the model invented cannot reach the user.
    """
    display = page_label(chunk)
    if chunk.section:
        display += ' - "{}"'.format(chunk.section)
    if include_document:
        display = "{}, {}".format(chunk.doc_title, display)
    return Citation(
        document=chunk.doc_title,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
        display=display,
    )


def _dedupe(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple] = set()
    out: list[Citation] = []
    for citation in citations:
        key = (citation.document, citation.page_start, citation.page_end, citation.section)
        if key not in seen:
            seen.add(key)
            out.append(citation)
    return out


def _trace_items(
    retrieved: list[RetrievedChunk], used: list[RetrievedChunk]
) -> list[RetrievalTraceItem]:
    used_ids = {item.chunk.chunk_id for item in used}
    return [
        RetrievalTraceItem(
            chunk_id=item.chunk.chunk_id,
            score=item.score,
            page_start=item.chunk.page_start,
            page_end=item.chunk.page_end,
            section=item.chunk.section,
            used_as_evidence=item.chunk.chunk_id in used_ids,
        )
        for item in retrieved
    ]


def answer_question(
    request: ChatRequest,
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    rewrite_llm: LLM,
    settings: Settings,
    rerank_llm: LLM | None = None,
    record=None,
) -> ChatResponse:
    started = time.perf_counter()
    # Every line logged inside this block carries the same id, so one question's
    # stages can be read together even with several requests in flight.
    with logs.request() as request_id:
        return _answer(
            request, store, embedder, llm, rewrite_llm, settings,
            rerank_llm, record, started, request_id,
        )


def _answer(
    request: ChatRequest,
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    rewrite_llm: LLM,
    settings: Settings,
    rerank_llm: LLM | None,
    record,
    started: float,
    request_id: str,
) -> ChatResponse:
    if request.doc_id is not None:
        store.get_document(request.doc_id)  # raises DocumentNotFound

    log.debug(
        "asked: %r%s", request.question[:120],
        " (in document {})".format(request.doc_id[:12]) if request.doc_id else "",
    )

    query, was_rewritten = rewrite_query(
        rewrite_llm, request.question, request.history, settings.max_history_turns
    )
    if was_rewritten:
        log.debug("rewrote follow-up to: %r", query[:120])

    # With reranking on we fetch a wider pool and let the model narrow it.
    wanted = settings.rerank_candidates if settings.rerank else settings.retriever_top_k
    retrieved = retrieve(
        store, embedder, query, wanted, request.doc_id,
        hybrid=settings.hybrid_search, rrf_k=settings.rrf_k,
    )
    log.debug(
        "retrieved %d/%d candidates by %s search%s",
        len(retrieved), wanted, "hybrid" if settings.hybrid_search else "dense",
        ", top score {:.3f}".format(retrieved[0].score) if retrieved else " (nothing)",
    )

    rerank_guards: list[str] = []
    if settings.rerank and retrieved:
        before = len(retrieved)
        retrieved, rerank_guards = rerank(
            rerank_llm or llm, query, retrieved, settings.retriever_top_k
        )
        log.debug("reranked %d candidates down to %d", before, len(retrieved))

    evidence, used = build_evidence(retrieved, settings.max_context_tokens)
    log.debug("evidence: %d chunks, about %d tokens", len(used), estimate_tokens(evidence))

    result = generate_answer(
        llm,
        request.question,
        request.history,
        evidence,
        len(used),
        settings.max_history_turns,
    )
    guards = result.guards + rerank_guards

    include_document = request.doc_id is None and len(store.documents) > 1
    citations = _dedupe(
        [build_citation(used[i - 1].chunk, include_document) for i in result.citations]
    )

    top_score = retrieved[0].score if retrieved else None
    entry = {
        "request_id": request_id,
        "doc_id": request.doc_id,
        "question": request.question[:200],
        "rewritten_query": query if was_rewritten else None,
        "retrieved": len(retrieved),
        "evidence": len(used),
        # Recorded on every request so the evaluation can compare score
        # distributions for answerable and unanswerable questions, and so the
        # dashboard can show what the system has been doing. See DESIGN.md.
        "top_score": top_score,
        "answerable": result.answerable,
        "citations": len(citations),
        "guards": guards,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
    # One line per question. Everything above it is DEBUG, so normal running
    # gives you exactly this, and turning on DEBUG explains any one of them.
    log.info(
        "%s | retrieved %d, evidence %d%s, citations %d | %dms%s",
        "answered" if result.answerable else "refused",
        len(retrieved),
        len(used),
        ", top {:.3f}".format(top_score) if top_score is not None else "",
        len(citations),
        entry["latency_ms"],
        " | guards: " + ", ".join(guards) if guards else "",
    )
    # The API stores this; the evaluation passes nothing, so a run does not fill
    # the dashboard with questions nobody asked.
    if record:
        record(entry)

    trace = None
    if request.debug:
        trace = Trace(
            request_id=request_id,
            rewritten_query=query if was_rewritten else None,
            retrieved=_trace_items(retrieved, used),
            top_score=top_score,
            evidence=evidence or None,
            raw_model_output=result.raw_output or None,
            guards=guards,
        )

    return ChatResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations=citations,
        trace=trace,
    )
