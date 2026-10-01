"""Build endpoints. Routes only translate HTTP ⇄ service calls."""

import json
from typing import Optional

from anyio import to_thread
from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from backend.app.core.dependencies import get_builds_service, rate_limit
from backend.app.core.security import Principal, require
from backend.app.modules.builds.api.schemas import (
    API_PREFIX, BuildCreated, BuildDetail, BuildList, CreateBuildRequest, build_detail, build_summary, event_out,
)
from backend.app.modules.builds.service import BuildsService
from metaagent.schemas import BuildStatus

router = APIRouter(prefix=f"{API_PREFIX}/builds", tags=["builds"])

_ERRORS = {401: {"description": "Missing or invalid API key"},
           403: {"description": "API key lacks permission"},
           404: {"description": "Build not found"}}


@router.post(
    "", status_code=202, response_model=BuildCreated,
    summary="Start a build",
    description=(
        "Queues a build and returns immediately with its `build_id`. The pipeline runs in the background: "
        "plan → select tools → generate → test → (self-correct and re-test on failure) → deploy, where "
        "deployment only happens if every test passed.\n\n"
        "Follow progress with `GET /api/v1/builds/{build_id}/events` (SSE) or poll "
        "`GET /api/v1/builds/{build_id}`.\n\n"
        "Requires the **builder** role: building runs LLM-generated code (in test mode) on this server."
    ),
    responses={**_ERRORS, 422: {"description": "Invalid request, unknown model, or request too long"},
               429: {"description": "Rate limit exceeded"}, 503: {"description": "Build queue is full"}},
)
def create_build(body: CreateBuildRequest,
                 principal: Principal = Depends(require("build")),
                 _: None = Depends(rate_limit("builds", "builds_per_hour", 3600)),
                 service: BuildsService = Depends(get_builds_service)) -> BuildCreated:
    record = service.create(body.request, body.constraints, body.model, body.use_cache, principal)
    return BuildCreated(build_id=record.id, status=record.status,
                        status_url=f"{API_PREFIX}/builds/{record.id}",
                        events_url=f"{API_PREFIX}/builds/{record.id}/events")


@router.get(
    "", response_model=BuildList, summary="List recent builds",
    description="Builds from persistent storage, newest first. Visible to every authenticated key.",
    responses=_ERRORS,
)
def list_builds(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                status: Optional[BuildStatus] = Query(None, description="Only builds in this status"),
                _: Principal = Depends(require("read")),
                service: BuildsService = Depends(get_builds_service)) -> BuildList:
    items = [build_summary(record, result) for record, result in service.list(limit, offset, status)]
    return BuildList(items=items, limit=limit, offset=offset)


@router.get(
    "/{build_id}", response_model=BuildDetail, summary="Get a build",
    description="Current persisted status and, once available, the plan, tool selection, generated code, "
                "test results and deployment of a build.",
    responses=_ERRORS,
)
def get_build(build_id: str, _: Principal = Depends(require("read")),
              service: BuildsService = Depends(get_builds_service)) -> BuildDetail:
    record, result = service.get(build_id)
    return build_detail(record, result)


@router.get(
    "/{build_id}/events", summary="Stream build events (SSE)",
    description=(
        "Server-Sent Events stream of a build's progress. Past events are replayed first, then live ones; "
        "the stream closes after the final `status` event (`succeeded` or `failed`).\n\n"
        "Each message has `id`, `event` (`stage` or `status`) and JSON `data` (see `BuildEventOut`). "
        "Stage events include `started`, `completed`, `failed`, `skipped`, `cache_hit`, `attempt_failed`, "
        "`self_correction` and `fallback`. Status events are `queued`, `running`, `succeeded`, `failed`.\n\n"
        "Reconnect with the `Last-Event-ID` header to resume. Comment lines (`: ping`) are heartbeats."
    ),
    response_class=StreamingResponse,
    responses={**_ERRORS, 200: {"content": {"text/event-stream": {}}, "description": "Event stream"}},
)
async def stream_build_events(build_id: str, request: Request,
                              last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
                              _: Principal = Depends(require("read")),
                              service: BuildsService = Depends(get_builds_service)) -> StreamingResponse:
    await to_thread.run_sync(service.get, build_id)  # 404 before the stream starts
    try:
        after_id = max(0, int(last_event_id)) if last_event_id else 0
    except ValueError:
        after_id = 0

    async def body():
        yield "retry: 3000\n\n"
        async for item in service.stream_events(build_id, after_id):
            if await request.is_disconnected():
                return
            if item is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event_out(item).model_dump(mode="json"))
            yield f"id: {item.id}\nevent: {item.kind}\ndata: {data}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
