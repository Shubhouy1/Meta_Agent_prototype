"""Central configuration for MetaAgent.

Every tunable value (models, timeouts, paths, limits) lives here. Values can be
overridden with environment variables; secrets (GOOGLE_API_KEY) are read from
the environment only and are never stored on the Settings object.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _env_list(name: str, default: tuple) -> tuple:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # Models
    default_model: str = "gemini-2.5-flash"
    available_models: tuple = ("gemini-2.5-flash", "gemini-2.5-pro")
    # Single source of truth for embeddings: ingestion, the RAG template, the
    # retriever snippet and the runtime proxy all use this value.
    embedding_model: str = "gemini-embedding-001"
    # Model the runtime proxy uses when a generated agent asks for one that is
    # not in available_models.
    agent_model: str = "gemini-2.5-flash"

    # LLM client behaviour
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    planner_temperature: float = 0.0
    selector_temperature: float = 0.0
    codegen_temperature: float = 0.3
    rag_temperature: float = 0.3

    # Pipeline limits
    max_request_chars: int = 2000
    max_generation_attempts: int = 2

    # Generated-code execution
    test_run_timeout_s: float = 60.0
    live_run_timeout_s: float = 120.0
    max_llm_calls_per_run: int = 5

    # RAG
    max_upload_mb: int = 20
    allowed_upload_types: tuple = ("pdf", "txt")
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 3

    # Runtime data (kept out of the source tree)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    @property
    def generated_dir(self) -> Path:
        return self.data_dir / "generated"

    @property
    def deployments_dir(self) -> Path:
        return self.data_dir / "deployments"

    @property
    def builds_dir(self) -> Path:
        return self.data_dir / "builds"

    @property
    def cache_file(self) -> Path:
        return self.data_dir / "cache" / "code_cache.json"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    """Build Settings from the environment (and .env, if present)."""
    load_dotenv(PROJECT_ROOT / ".env")
    defaults = Settings()
    available = _env_list("METAAGENT_AVAILABLE_MODELS", defaults.available_models)
    default_model = _env_str("METAAGENT_DEFAULT_MODEL", defaults.default_model)
    if default_model not in available:
        available = (default_model,) + available
    return Settings(
        default_model=default_model,
        available_models=available,
        embedding_model=_env_str("METAAGENT_EMBEDDING_MODEL", defaults.embedding_model),
        agent_model=_env_str("METAAGENT_AGENT_MODEL", defaults.agent_model),
        llm_timeout_s=_env_float("METAAGENT_LLM_TIMEOUT_S", defaults.llm_timeout_s),
        llm_max_retries=_env_int("METAAGENT_LLM_MAX_RETRIES", defaults.llm_max_retries),
        max_request_chars=_env_int("METAAGENT_MAX_REQUEST_CHARS", defaults.max_request_chars),
        max_generation_attempts=_env_int("METAAGENT_MAX_GENERATION_ATTEMPTS", defaults.max_generation_attempts),
        test_run_timeout_s=_env_float("METAAGENT_TEST_RUN_TIMEOUT_S", defaults.test_run_timeout_s),
        live_run_timeout_s=_env_float("METAAGENT_LIVE_RUN_TIMEOUT_S", defaults.live_run_timeout_s),
        max_llm_calls_per_run=_env_int("METAAGENT_MAX_LLM_CALLS_PER_RUN", defaults.max_llm_calls_per_run),
        max_upload_mb=_env_int("METAAGENT_MAX_UPLOAD_MB", defaults.max_upload_mb),
        rag_top_k=_env_int("METAAGENT_RAG_TOP_K", defaults.rag_top_k),
        data_dir=Path(_env_str("METAAGENT_DATA_DIR", str(defaults.data_dir))),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
