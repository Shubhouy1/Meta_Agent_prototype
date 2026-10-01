"""API-level settings.

Reuses the engine's configuration mechanism (metaagent.core.config: same
.env file, same env helpers) and holds the engine Settings instead of
duplicating them. Only concerns that exist because of the HTTP API live here.

Secrets (API keys) come from the environment only.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

from metaagent.core.config import Settings, env_bool, env_float, env_int, env_list, env_str, load_settings

logger = logging.getLogger(__name__)

ROLES = ("viewer", "builder")
MIN_API_KEY_LENGTH = 24


@dataclass(frozen=True)
class ApiKey:
    name: str
    role: str
    key: str = field(repr=False)


def parse_api_keys(raw: str) -> Tuple[ApiKey, ...]:
    """Parse "name:role:key;name2:role2:key2". Invalid entries are skipped with a warning."""
    keys = []
    for entry in (raw or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 2)
        if len(parts) != 3:
            logger.warning("Ignoring malformed METAAGENT_API_KEYS entry (expected name:role:key)")
            continue
        name, role, key = (p.strip() for p in parts)
        if role not in ROLES:
            logger.warning("Ignoring API key %r: unknown role %r (allowed: %s)", name, role, ", ".join(ROLES))
            continue
        if len(key) < MIN_API_KEY_LENGTH:
            logger.warning("Ignoring API key %r: shorter than %d characters", name, MIN_API_KEY_LENGTH)
            continue
        keys.append(ApiKey(name=name, role=role, key=key))
    return tuple(keys)


@dataclass(frozen=True)
class ApiSettings:
    engine: Settings = field(default_factory=Settings)
    database_url: str = ""
    db_auto_migrate: bool = True

    api_keys: Tuple[ApiKey, ...] = ()
    cors_origins: Tuple[str, ...] = ("http://localhost:3000",)
    docs_enabled: bool = True

    # Background builds
    max_concurrent_builds: int = 2
    max_queued_builds: int = 10

    # Rate limits per API key (requests per window)
    builds_per_hour: int = 20
    invocations_per_minute: int = 10
    rag_queries_per_minute: int = 20
    rag_uploads_per_hour: int = 30
    max_concurrent_invocations: int = 2

    # Request limits
    max_body_bytes: int = 1024 * 1024          # JSON endpoints
    max_upload_files: int = 5

    # SSE
    sse_heartbeat_s: float = 15.0

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.engine.data_dir / 'metaagent.db').as_posix()}"

    @property
    def max_upload_request_bytes(self) -> int:
        # All files at the per-file limit plus multipart overhead.
        return self.engine.max_upload_bytes * self.max_upload_files + 64 * 1024

    def key_index(self) -> Dict[str, ApiKey]:
        return {k.key: k for k in self.api_keys}


def load_api_settings(engine: Optional[Settings] = None) -> ApiSettings:
    engine = engine or load_settings()  # also loads .env
    defaults = ApiSettings(engine=engine)
    return ApiSettings(
        engine=engine,
        database_url=env_str("METAAGENT_DATABASE_URL", ""),
        db_auto_migrate=env_bool("METAAGENT_DB_AUTO_MIGRATE", True),
        api_keys=parse_api_keys(env_str("METAAGENT_API_KEYS", "")),
        cors_origins=env_list("METAAGENT_CORS_ORIGINS", defaults.cors_origins),
        docs_enabled=env_bool("METAAGENT_DOCS_ENABLED", True),
        max_concurrent_builds=env_int("METAAGENT_MAX_CONCURRENT_BUILDS", defaults.max_concurrent_builds),
        max_queued_builds=env_int("METAAGENT_MAX_QUEUED_BUILDS", defaults.max_queued_builds),
        builds_per_hour=env_int("METAAGENT_BUILDS_PER_HOUR", defaults.builds_per_hour),
        invocations_per_minute=env_int("METAAGENT_INVOCATIONS_PER_MINUTE", defaults.invocations_per_minute),
        rag_queries_per_minute=env_int("METAAGENT_RAG_QUERIES_PER_MINUTE", defaults.rag_queries_per_minute),
        rag_uploads_per_hour=env_int("METAAGENT_RAG_UPLOADS_PER_HOUR", defaults.rag_uploads_per_hour),
        max_concurrent_invocations=env_int("METAAGENT_MAX_CONCURRENT_INVOCATIONS",
                                           defaults.max_concurrent_invocations),
        max_body_bytes=env_int("METAAGENT_MAX_BODY_BYTES", defaults.max_body_bytes),
        sse_heartbeat_s=env_float("METAAGENT_SSE_HEARTBEAT_S", defaults.sse_heartbeat_s),
    )


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return load_api_settings()
