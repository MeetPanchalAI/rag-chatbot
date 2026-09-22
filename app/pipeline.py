"""The chat flow: rewrite, retrieve, generate, cite.

Kept out of the API layer so the whole flow can be tested without HTTP, and so
the evaluation script runs the exact same code path as a real request.
"""

import json
import logging
import time
import uuid

from app.config import Settings
from app.generation import generate_answer
from app.providers import Embedder, LLM
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
from app.vector_store import VectorStore

log = logging.getLogger("rag.chat")


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
) -> ChatResponse:
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    if request.doc_id is not None:
        store.get_document(request.doc_id)  # raises DocumentNotFound

    query, was_rewritten = rewrite_query(
        rewrite_llm, request.question, request.history, settings.max_history_turns
    )
    retrieved = retrieve(
        store, embedder, query, settings.retriever_top_k, request.doc_id
    )
    evidence, used = build_evidence(retrieved, settings.max_context_tokens)

    result = generate_answer(
        llm,
        request.question,
        request.history,
        evidence,
        len(used),
        settings.max_history_turns,
    )

    include_document = request.doc_id is None and len(store.documents) > 1
    citations = _dedupe(
        [build_citation(used[i - 1].chunk, include_document) for i in result.citations]
    )

    top_score = retrieved[0].score if retrieved else None
    log.info(
        json.dumps(
            {
                "request_id": request_id,
                "doc_id": request.doc_id,
                "question": request.question[:200],
                "rewritten_query": query if was_rewritten else None,
                "retrieved": len(retrieved),
                "evidence": len(used),
                # Logged on every request so the evaluation run can compare score
                # distributions for answerable and unanswerable questions, and
                # show whether a relevance threshold would help. See DESIGN.md.
                "top_score": top_score,
                "answerable": result.answerable,
                "citations": len(citations),
                "guards": result.guards,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        )
    )

    trace = None
    if request.debug:
        trace = Trace(
            request_id=request_id,
            rewritten_query=query if was_rewritten else None,
            retrieved=_trace_items(retrieved, used),
            top_score=top_score,
            evidence=evidence or None,
            raw_model_output=result.raw_output or None,
            guards=result.guards,
        )

    return ChatResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations=citations,
        trace=trace,
    )
