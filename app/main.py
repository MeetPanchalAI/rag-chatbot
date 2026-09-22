"""HTTP API.

This layer only validates input, wires dependencies and maps errors. The actual
work lives in ingestion, retrieval, generation and pipeline.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import AppError, FileTooLarge, UnsupportedFile
from app.indexer import index_pdf
from app.pipeline import answer_question
from app.providers import Embedder, LLM, OpenAIEmbedder, OpenAILLM
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    HealthResponse,
    IngestResponse,
)
from app.vector_store import VectorStore

log = logging.getLogger(__name__)


class Providers:
    """Builds providers on first use, so the app starts without an API key."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._embedder: Embedder | None = None
        self._llm: LLM | None = None
        self._rewrite_llm: LLM | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(self._settings)
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = OpenAILLM(self._settings)
        return self._llm

    @property
    def rewrite_llm(self) -> LLM:
        if self._rewrite_llm is None:
            self._rewrite_llm = OpenAILLM(
                self._settings, model=self._settings.rewrite_model_name
            )
        return self._rewrite_llm


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    providers: Providers | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    uses_real_providers = providers is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail at startup rather than on the first question, so a missing key is
        # a boot error with a clear message instead of a 502 later.
        if uses_real_providers and not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        app.state.store.load()
        yield

    app = FastAPI(
        title="RAG Document Chatbot",
        description="Ask questions about an ingested PDF and get answers with citations.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store or VectorStore(settings.data_dir)
    app.state.providers = providers or Providers(settings)

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            documents=len(app.state.store.documents),
            chunks=app.state.store.chunk_count,
        )

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents() -> list[DocumentInfo]:
        return app.state.store.list_documents()

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(file: UploadFile = File(...)) -> IngestResponse:
        name = file.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            raise UnsupportedFile("Only PDF files are accepted.")

        data = await file.read()
        limit = settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise FileTooLarge(
                "The file is larger than {} MB. Use the ingest script for large "
                "documents: python -m app.ingest_cli <path>".format(settings.max_upload_mb)
            )

        return index_pdf(
            data, name, app.state.store, app.state.providers.embedder, settings
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return answer_question(
            request,
            app.state.store,
            app.state.providers.embedder,
            app.state.providers.llm,
            app.state.providers.rewrite_llm,
            settings,
        )

    return app


app = create_app()
