"""Composition root: builds every service once per app and hands them to
routes through FastAPI dependencies. Tests swap in fakes by passing
overrides to create_app() (fake LLM, fake embeddings), not by patching."""

from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, Request
from sqlalchemy.engine import Engine

from backend.app.core.config import ApiSettings
from backend.app.core.rate_limit import ConcurrencyGate, RateLimiter
from backend.app.core.security import Principal, current_principal
from backend.app.db.build_store import SqlBuildStore, SqlEventStore
from backend.app.db.session import make_engine
from backend.app.modules.builds.events import EventBroker
from backend.app.modules.builds.jobs import JobManager
from backend.app.modules.builds.service import BuildsService
from backend.app.modules.deployments.service import DeploymentsService
from backend.app.modules.rag.service import RagService
from metaagent.builds.service import BuildService
from metaagent.execution.runner import AgentRunner
from metaagent.rag.service import DocumentService

DocumentServiceFactory = Callable[[str], DocumentService]


@dataclass
class Container:
    settings: ApiSettings
    db: Engine
    build_store: SqlBuildStore
    broker: EventBroker
    jobs: JobManager
    rate_limiter: RateLimiter
    builds: BuildsService
    deployments: DeploymentsService
    rag: RagService

    def close(self) -> None:
        self.jobs.shutdown(wait=False)
        self.db.dispose()


def build_container(settings: ApiSettings, *, llm=None, runner: Optional[AgentRunner] = None,
                    document_service_factory: Optional[DocumentServiceFactory] = None) -> Container:
    db = make_engine(settings.resolved_database_url)
    build_store = SqlBuildStore(db)
    broker = EventBroker(SqlEventStore(db))
    jobs = JobManager(settings.max_concurrent_builds, settings.max_queued_builds)
    engine_builds = BuildService(settings.engine, store=build_store,
                                 runner=runner or AgentRunner(settings.engine), llm=llm)
    factory = document_service_factory or (lambda collection: DocumentService(settings.engine,
                                                                            collection=collection))
    return Container(
        settings=settings, db=db, build_store=build_store, broker=broker, jobs=jobs,
        rate_limiter=RateLimiter(),
        builds=BuildsService(settings, engine_builds, build_store, broker, jobs),
        deployments=DeploymentsService(engine_builds, build_store,
                                       ConcurrencyGate(settings.max_concurrent_invocations, "invocations")),
        rag=RagService(factory),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_builds_service(request: Request) -> BuildsService:
    return get_container(request).builds


def get_deployments_service(request: Request) -> DeploymentsService:
    return get_container(request).deployments


def get_rag_service(request: Request) -> RagService:
    return get_container(request).rag


def rate_limit(action: str, limit_setting: str, window_s: float):
    """Dependency factory: count one request against the caller's limit for `action`."""

    def dependency(request: Request, principal: Principal = Depends(current_principal)) -> None:
        container = get_container(request)
        container.rate_limiter.check(principal.name, action, getattr(container.settings, limit_setting), window_s)

    dependency.__name__ = f"rate_limit_{action}"
    return dependency
