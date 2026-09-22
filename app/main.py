"""HTTP API and the browser UI.

This layer validates input, wires dependencies and maps errors. The work lives
in ingestion, retrieval, generation, pipeline and evaluation.
"""

import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import conversations as convo
from app import eval_runs
from app.config import Settings, get_settings
from app.db import Database
from app.errors import AppError, EvaluationRunning, FileTooLarge, UnsupportedFile
from app.evaluation import (
    breakdown,
    headline,
    load_questions,
    run_evaluation,
    score_ranges,
    summarise,
    write_report,
)
from app.indexer import index_pdf
from app.pipeline import answer_question
from app.providers import Embedder, LLM, OpenAIEmbedder, OpenAILLM
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Conversation,
    DocumentInfo,
    EvalRequest,
    EvalRunInfo,
    EvalStatus,
    HealthResponse,
    IngestResponse,
    NewConversation,
)
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Providers:
    """Builds providers on first use, so the app starts without an API key."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._embedder: Embedder | None = None
        self._llm: LLM | None = None
        self._rewrite_llm: LLM | None = None
        self._judge_llm: LLM | None = None

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

    @property
    def judge_llm(self) -> LLM:
        if self._judge_llm is None:
            self._judge_llm = OpenAILLM(
                self._settings, model=self._settings.judge_model_name
            )
        return self._judge_llm


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
    db = store.db if store else Database(settings.data_dir / "app.db")
    start_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail at startup rather than on the first question, so a missing key is
        # a boot error with a clear message instead of a 502 later.
        if uses_real_providers and not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
            )
        app.state.store.load()
        # A run that was in flight when the process died is not still running.
        with db.write() as connection:
            connection.execute(
                "UPDATE eval_runs SET status = 'failed', "
                "error = 'interrupted by a restart' WHERE status = 'running'"
            )
        yield

    app = FastAPI(
        title="RAG Document Chatbot",
        description="Ask questions about an ingested PDF and get answers with citations.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.store = store or VectorStore(db, settings.data_dir)
    app.state.providers = providers or Providers(settings)

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    # --- UI ---

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # --- status and documents ---

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

    @app.delete("/documents/{doc_id}", response_model=DocumentInfo)
    def remove_document(doc_id: str) -> DocumentInfo:
        return app.state.store.delete_document(doc_id)

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

    # --- conversations ---

    @app.get("/conversations", response_model=list[Conversation])
    def list_conversations() -> list[Conversation]:
        return convo.list_all(db)

    @app.post("/conversations", response_model=Conversation)
    def new_conversation(request: NewConversation) -> Conversation:
        return convo.create(db, request.title, request.doc_id)

    @app.get("/conversations/{conversation_id}", response_model=Conversation)
    def read_conversation(conversation_id: str) -> Conversation:
        return convo.get(db, conversation_id)

    @app.delete("/conversations/{conversation_id}")
    def drop_conversation(conversation_id: str) -> dict:
        convo.delete(db, conversation_id)
        return {"deleted": conversation_id}

    # --- chat ---

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        # With a conversation_id the server owns the history; without one the
        # caller supplies it and nothing is written. The evaluation uses the
        # stateless path, so its results never depend on what is stored.
        if request.conversation_id:
            convo.get(db, request.conversation_id)
            request = request.model_copy(
                update={
                    "history": convo.history(
                        db, request.conversation_id, settings.max_history_turns
                    )
                }
            )

        response = answer_question(
            request,
            app.state.store,
            app.state.providers.embedder,
            app.state.providers.llm,
            app.state.providers.rewrite_llm,
            settings,
        )

        if request.conversation_id:
            convo.append(db, request.conversation_id, "user", request.question)
            convo.append(
                db, request.conversation_id, "assistant", response.answer,
                answerable=response.answerable, citations=response.citations,
                trace=response.trace,
            )
        return response

    # --- evaluation ---

    @app.get("/eval/status", response_model=EvalStatus)
    def eval_status() -> EvalStatus:
        return eval_runs.latest(db) or EvalStatus()

    @app.get("/eval/runs", response_model=list[EvalRunInfo])
    def eval_history() -> list[EvalRunInfo]:
        return eval_runs.list_runs(db)

    @app.get("/eval/runs/{run_id}", response_model=EvalStatus)
    def eval_run(run_id: int) -> EvalStatus:
        return eval_runs.get_run(db, run_id)

    @app.delete("/eval/runs/{run_id}")
    def drop_eval_run(run_id: int) -> dict:
        eval_runs.delete(db, run_id)
        return {"deleted": run_id}

    @app.post("/eval/run", response_model=EvalStatus)
    def eval_start(request: EvalRequest) -> EvalStatus:
        store_ = app.state.store
        providers_ = app.state.providers

        if request.doc_id is not None:
            store_.get_document(request.doc_id)
        questions = load_questions()

        with start_lock:
            if db.one("SELECT 1 FROM eval_runs WHERE status = 'running' LIMIT 1"):
                raise EvaluationRunning("An evaluation is already running.")
            run_id = eval_runs.start(
                db, len(questions), request.doc_id, settings, request.label
            )

        def work() -> None:
            try:
                rows = run_evaluation(
                    questions, store_, providers_.embedder, providers_.llm,
                    providers_.rewrite_llm, settings, doc_id=request.doc_id,
                    judge_llm=getattr(providers_, "judge_llm", None) if request.judge else None,
                    on_progress=lambda done, _total: eval_runs.progress(db, run_id, done),
                )
                eval_runs.finish(
                    db, run_id, rows, summarise(rows), headline(rows),
                    score_ranges(rows), breakdown(rows),
                )
                out_dir = settings.eval_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                write_report(rows, out_dir / "results.md")
                (out_dir / "results.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
                )
            except Exception as exc:  # the UI shows this; a run must not vanish
                log.exception("Evaluation failed")
                eval_runs.fail(db, run_id, str(exc))

        threading.Thread(target=work, name="evaluation", daemon=True).start()
        return eval_runs.get_run(db, run_id)

    return app


app = create_app()
