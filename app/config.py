"""Application settings. Everything comes from the environment or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # Models
    llm_model: str = "gpt-5.6-luna"
    rewrite_model: str = ""
    judge_model: str = ""
    rerank_model: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 1.0
    # Blank falls back to LLM_TEMPERATURE.
    judge_temperature: float | None = None
    llm_reasoning_effort: str = "low"

    # Storage
    data_dir: Path = Path("./data")
    # Where evaluation reports are written
    eval_dir: Path = Path("./eval")
    # Keep the uploaded PDF so the corpus can be re-chunked without it being re-supplied
    keep_source_pdf: bool = True

    # Chunking
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 75

    # Retrieval
    retriever_top_k: int = 6
    max_context_tokens: int = 4000
    # Keyword search fused with the dense search. Costs no extra API call.
    hybrid_search: bool = True
    rrf_k: int = 60
    # Reranking costs one model call per question, so it is opt-in.
    rerank: bool = False
    rerank_candidates: int = 20

    # Limits
    max_history_turns: int = 3
    max_question_chars: int = 2000
    max_upload_mb: int = 50
    embed_batch_size: int = 128
    request_timeout_s: float = 60.0

    # Logging
    log_level: str = "INFO"

    @property
    def rewrite_model_name(self) -> str:
        return self.rewrite_model or self.llm_model

    @property
    def judge_model_name(self) -> str:
        return self.judge_model or self.llm_model

    @property
    def rerank_model_name(self) -> str:
        return self.rerank_model or self.llm_model

    @property
    def judge_temperature_value(self) -> float:
        return self.llm_temperature if self.judge_temperature is None else self.judge_temperature


@lru_cache
def get_settings() -> Settings:
    return Settings()
