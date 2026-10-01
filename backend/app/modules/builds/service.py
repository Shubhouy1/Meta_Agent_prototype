"""Builds application service.

    route → BuildsService → metaagent BuildService.start_build → pipeline.run_build

Owns the asynchronous lifecycle the engine doesn't need to know about:
validate → persist as "queued" → run on the job manager → "running" →
engine pipeline (events via on_event) → final result persisted by the engine
BuildService through SqlBuildStore → terminal status event.
"""

import asyncio
import logging
from typing import AsyncIterator, List, Optional, Tuple

from anyio import to_thread

from backend.app.core.config import ApiSettings
from backend.app.core.errors import InvalidRequest, NotFound, ServiceUnavailable
from backend.app.core.security import Principal
from backend.app.db.build_store import SqlBuildStore
from backend.app.db.models import BuildRecord
from backend.app.modules.builds.events import EventBroker, StreamEvent
from backend.app.modules.builds.jobs import JobManager, QueueFull
from metaagent.builds.service import BuildService
from metaagent.pipeline import new_build_id
from metaagent.schemas import BuildResult, Constraints, StageError

logger = logging.getLogger(__name__)


class BuildsService:
    def __init__(self, settings: ApiSettings, engine: BuildService, store: SqlBuildStore,
                 broker: EventBroker, jobs: JobManager):
        self.settings = settings
        self.engine = engine
        self.store = store
        self.broker = broker
        self.jobs = jobs

    # ── commands ──

    def create(self, request: str, constraints: Constraints, model: Optional[str],
               use_cache: bool, principal: Principal) -> BuildRecord:
        engine_settings = self.settings.engine
        request = request.strip()
        if not request:
            raise InvalidRequest("Describe the agent you want to build.")
        if len(request) > engine_settings.max_request_chars:
            raise InvalidRequest(f"Request is too long ({len(request)} characters, "
                                 f"max {engine_settings.max_request_chars}).")
        model = model or engine_settings.default_model
        if model not in engine_settings.available_models:
            raise InvalidRequest(f"Model '{model}' is not enabled. "
                                 f"Available: {', '.join(engine_settings.available_models)}")
        if not self.jobs.has_capacity():
            raise ServiceUnavailable("The build queue is full; try again later.", headers={"Retry-After": "30"})

        build_id = new_build_id()
        queued = BuildResult(build_id=build_id, request=request, model=model,
                             constraints=constraints, status="queued")
        record = self.store.create(queued, owner=principal.name)
        self.broker.publish(build_id, "status", "queued", message="Build queued")
        try:
            self.jobs.submit(build_id, lambda: self._execute(build_id, request, constraints, model, use_cache))
        except QueueFull as e:
            queued.status = "failed"
            queued.error = StageError(type="QueueFull", message="The build queue is full; try again later.")
            self.store.save(queued)
            self.broker.publish(build_id, "status", "failed", message=queued.error.message)
            raise ServiceUnavailable(queued.error.message, headers={"Retry-After": "30"}) from e
        logger.info("Build %s queued by %s", build_id, principal.name)
        return record

    def _execute(self, build_id: str, request: str, constraints: Constraints, model: str,
                 use_cache: bool) -> None:
        self.store.mark_running(build_id)
        self.broker.publish(build_id, "status", "running", message="Build started")
        try:
            result = self.engine.start_build(request, constraints, model, use_cache=use_cache,
                                             build_id=build_id, on_event=self.broker.publish_pipeline_event)
        except Exception as e:  # run_build never raises for pipeline failures; this is a last resort
            logger.exception("Build %s crashed", build_id)
            result = BuildResult(build_id=build_id, request=request, model=model, constraints=constraints,
                                 status="failed", error=StageError(type=type(e).__name__,
                                                                   message="Internal error while building"))
            self.store.save(result)
        message = ("Agent deployed" if result.deployed else
                   result.error.message if result.error else result.status)
        self.broker.publish(build_id, "status", result.status, stage=result.failed_stage, message=message)

    def recover_interrupted(self) -> List[str]:
        """Called at startup: builds a previous process never finished are marked failed."""
        ids = self.store.fail_interrupted()
        for build_id in ids:
            self.broker.publish(build_id, "status", "failed",
                                message="The server stopped before this build finished.")
        if ids:
            logger.warning("Marked %d interrupted build(s) as failed", len(ids))
        return ids

    # ── queries ──

    def get(self, build_id: str) -> Tuple[BuildRecord, BuildResult]:
        record = self.store.get_record(build_id)
        if record is None:
            raise NotFound("Build not found")
        return record, BuildResult.model_validate_json(record.result_json)

    def list(self, limit: int, offset: int, status: Optional[str]) -> List[Tuple[BuildRecord, BuildResult]]:
        return [(r, BuildResult.model_validate_json(r.result_json))
                for r in self.store.list_records(limit=limit, offset=offset, status=status)]

    async def stream_events(self, build_id: str, after_id: int = 0) -> AsyncIterator[Optional[StreamEvent]]:
        """Yield stored events after `after_id`, then live ones, ending after the
        terminal status event. Yields None when a heartbeat is due."""
        subscription = self.broker.subscribe(build_id)  # subscribe first so nothing is missed
        try:
            # A build's last event is always its terminal status event, so the
            # stream ends on events, never on the record's status (the engine
            # saves the final record slightly before the terminal event exists).
            history = await to_thread.run_sync(self.broker.history, build_id, 0)
            last_id = after_id
            for item in history:
                if item.id <= after_id:
                    if item.terminal:
                        return  # the client already received the end of this build
                    continue
                last_id = item.id
                yield item
                if item.terminal:
                    return
            idle_beats = 0
            while True:
                try:
                    item = await asyncio.wait_for(subscription.queue.get(), self.settings.sse_heartbeat_s)
                except asyncio.TimeoutError:
                    yield None
                    idle_beats += 1
                    if idle_beats >= 2 and await self._finished_without_terminal_event(build_id):
                        return
                    continue
                idle_beats = 0
                if item.id <= last_id:
                    continue
                last_id = item.id
                yield item
                if item.terminal:
                    return
        finally:
            self.broker.unsubscribe(subscription)

    async def _finished_without_terminal_event(self, build_id: str) -> bool:
        """Safety net for builds finished outside the API job path (no status events)."""
        record = await to_thread.run_sync(self.store.get_record, build_id)
        if record is None or record.status not in ("succeeded", "failed"):
            return False
        history = await to_thread.run_sync(self.broker.history, build_id, 0)
        return not any(e.terminal for e in history)
