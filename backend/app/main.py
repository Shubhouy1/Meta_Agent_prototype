"""MetaAgent HTTP API.

    uvicorn backend.app.main:app --reload

Dependency direction: backend → metaagent. Nothing in metaagent/ imports
FastAPI or this package.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import ApiSettings, load_api_settings
from backend.app.core.dependencies import DocumentServiceFactory, build_container
from backend.app.core.errors import install_error_handlers
from backend.app.core.middleware import BodySizeLimitMiddleware
from backend.app.db.session import run_migrations
from backend.app.modules.builds.api.routes import router as builds_router
from backend.app.modules.deployments.api.routes import router as deployments_router
from backend.app.modules.rag.api.routes import router as rag_router
from metaagent.core.logging import configure_logging
from metaagent.execution.runner import AgentRunner

logger = logging.getLogger("metaagent.api")

API_VERSION = "0.2.0"

DESCRIPTION = """
MetaAgent turns a plain-English description into a planned, generated, **tested** and deployed AI agent.

### Flow
1. `POST /api/v1/builds` queues a build and returns a `build_id` (HTTP 202).
2. `GET /api/v1/builds/{build_id}/events` streams progress as Server-Sent Events, or poll
   `GET /api/v1/builds/{build_id}`.
3. The pipeline plans, selects tools, generates code, tests it (self-correcting and re-testing on failure)
   and deploys **only if every test passed**.
4. `POST /api/v1/deployments/{deployment_id}/invoke` runs the deployed agent.

### Authentication
Send `Authorization: Bearer <api key>`. Keys are configured on the server in `METAAGENT_API_KEYS`.
* **viewer**: read builds, events, deployments and the document list.
* **builder**: additionally create builds, invoke deployments, upload/delete documents and query them.

### Safety
Generated agents run in a separate process with a timeout and without API keys, but that runner is
**not a hardened sandbox**. Only give builder keys to people you trust to run code on this server.

### Errors
Every error response is `{"detail": "..."}` (validation errors also include `errors`).
"""

TAGS = [
    {"name": "builds", "description": "Create builds, follow their progress, browse history."},
    {"name": "deployments", "description": "Inspect and invoke agents that passed testing."},
    {"name": "rag", "description": "Shared knowledge base used by RAG agents: upload, list, delete, query."},
    {"name": "system", "description": "Health check."},
]


def create_app(settings: Optional[ApiSettings] = None, *, llm=None, runner: Optional[AgentRunner] = None,
               document_service_factory: Optional[DocumentServiceFactory] = None) -> FastAPI:
    """Application factory. The keyword overrides exist for tests (fake LLM / embeddings)."""
    settings = settings or load_api_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.db_auto_migrate:
            run_migrations(settings.resolved_database_url)
        container = build_container(settings, llm=llm, runner=runner,
                                    document_service_factory=document_service_factory)
        app.state.container = container
        container.builds.recover_interrupted()
        if not settings.api_keys:
            logger.warning("No valid API keys configured (METAAGENT_API_KEYS); every protected endpoint "
                           "will return 401")
        try:
            yield
        finally:
            container.close()

    app = FastAPI(
        title="MetaAgent API", version=API_VERSION, description=DESCRIPTION, openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    install_error_handlers(app)

    upload_path = "/api/v1/rag/documents"
    app.add_middleware(
        BodySizeLimitMiddleware, default_limit=settings.max_body_bytes,
        limit_for_path=lambda path: settings.max_upload_request_bytes if path == upload_path else settings.max_body_bytes,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
        expose_headers=["Retry-After"],
    )

    app.include_router(builds_router)
    app.include_router(deployments_router)
    app.include_router(rag_router)

    @app.get("/health", tags=["system"], summary="Liveness check",
             description="Public. Reports only that the process is up; no configuration or data.")
    def health() -> dict:
        return {"status": "ok", "version": API_VERSION}

    return app


app = create_app()
