"""HTTP schemas for builds.

These are client-facing *views* of the engine's BuildResult. Engine models
are reused directly where they carry nothing sensitive (Constraints, Plan,
ToolChoice, ToolSimulation). Everything else is projected so that absolute
filesystem paths, tracebacks and raw provider payloads never leave the
server. The full record stays in the database.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.errors import redact
from backend.app.db.models import BuildRecord
from backend.app.modules.builds.events import StreamEvent
from metaagent.schemas import (
    BuildResult, BuildStatus, Constraints, GenerationAttempt, Plan, TestCheck, ToolChoice, ToolSimulation,
)

API_PREFIX = "/api/v1"


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ── requests ───────────────────────────────────────────

class CreateBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "request": "Build a calculator that can do basic math",
        "model": "gemini-2.5-flash",
        "constraints": {"budget": "free", "privacy": "strict", "performance": "balanced"},
        "use_cache": True,
    }]})

    request: str = Field(min_length=1, max_length=10_000,
                         description="Plain-English description of the agent to build. "
                                     "The server enforces its configured maximum length.")
    model: Optional[str] = Field(default=None, description="Model for planning, tool selection and code "
                                                           "generation. Defaults to the server's default model.")
    constraints: Constraints = Field(default_factory=Constraints,
                                     description="Tool selection constraints used by the what-if simulation.")
    use_cache: bool = Field(default=True, description="Reuse a cached generation if it still passes testing.")


# ── responses ──────────────────────────────────────────

class ErrorInfo(BaseModel):
    type: str
    message: str


class BuildCreated(BaseModel):
    build_id: str
    status: BuildStatus
    status_url: str
    events_url: str


class BuildSummary(BaseModel):
    build_id: str
    status: BuildStatus
    request: str
    model: Optional[str]
    agent_type: Optional[str]
    failed_stage: Optional[str]
    error: Optional[ErrorInfo]
    deployed: bool
    deployment_id: Optional[str]
    owner: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    duration_s: Optional[float]


class PlanView(BaseModel):
    plan: Plan
    fallback_used: bool
    warnings: List[str]


class ToolSelectionView(BaseModel):
    selection_method: str
    selections: Dict[str, ToolChoice]
    simulations: Dict[str, ToolSimulation]
    warnings: List[str]
    error: Optional[ErrorInfo]


class GenerationView(BaseModel):
    method: Optional[str]
    model: Optional[str]
    lines: int
    quality_score: float
    corrections: int
    attempts: List[GenerationAttempt]
    code: Optional[str]
    error: Optional[ErrorInfo]


class TestView(BaseModel):
    __test__ = False

    passed: bool
    checks: List[TestCheck]
    warnings: List[str]
    output_preview: Optional[str]
    execution_time_s: Optional[float]
    llm_calls: int
    error: Optional[ErrorInfo]


class DeploymentRef(BaseModel):
    deployment_id: str
    agent_type: Optional[str]
    url: str
    invoke_url: str


class BuildDetail(BuildSummary):
    constraints: Constraints
    plan: Optional[PlanView]
    tool_selection: Optional[ToolSelectionView]
    generation: Optional[GenerationView]
    test: Optional[TestView]
    deployment: Optional[DeploymentRef]


class BuildList(BaseModel):
    items: List[BuildSummary]
    limit: int
    offset: int


class BuildEventOut(BaseModel):
    id: int = Field(description="Monotonic event id; send it back as Last-Event-ID to resume a stream")
    build_id: str
    kind: Literal["stage", "status"]
    event: str
    stage: Optional[str]
    message: str
    attempt: Optional[int]
    created_at: datetime


# ── projections ────────────────────────────────────────

def _error(err) -> Optional[ErrorInfo]:
    return ErrorInfo(type=err.type, message=redact(err.message)) if err else None


def deployment_ref(build_id: str, result: BuildResult) -> Optional[DeploymentRef]:
    if not result.deployed:
        return None
    dep_id = result.deployment.deployment_id or build_id
    return DeploymentRef(deployment_id=dep_id, agent_type=result.deployment.agent_type,
                         url=f"{API_PREFIX}/deployments/{dep_id}",
                         invoke_url=f"{API_PREFIX}/deployments/{dep_id}/invoke")


def build_summary(record: BuildRecord, result: BuildResult) -> BuildSummary:
    return BuildSummary(
        build_id=record.id, status=record.status, request=record.request, model=record.model,
        agent_type=record.agent_type, failed_stage=record.failed_stage, error=_error(result.error),
        deployed=record.deployed,
        deployment_id=(result.deployment.deployment_id if result.deployed else None),
        owner=record.owner, created_at=as_utc(record.created_at), started_at=as_utc(record.started_at),
        finished_at=as_utc(record.finished_at), duration_s=record.duration_s,
    )


def build_detail(record: BuildRecord, result: BuildResult) -> BuildDetail:
    plan = tool = gen = test = None
    if result.plan is not None and result.plan.ok:
        plan = PlanView(plan=result.plan.plan, fallback_used=result.plan.fallback_used,
                        warnings=[redact(w) for w in result.plan.warnings])
    if result.tool_selection is not None:
        ts = result.tool_selection
        tool = ToolSelectionView(selection_method=ts.selection_method, selections=ts.selections,
                                 simulations=ts.simulations, warnings=[redact(w) for w in ts.warnings],
                                 error=_error(ts.error))
    if result.generation is not None:
        g = result.generation
        gen = GenerationView(
            method=g.method, model=g.model, lines=g.lines, quality_score=g.quality_score,
            corrections=g.corrections, code=g.code, error=_error(g.error),
            attempts=[a.model_copy(update={"error": redact(a.error)}) for a in g.attempts],
        )
    if result.test is not None:
        t = result.test
        test = TestView(
            passed=t.passed, warnings=[redact(w) for w in t.warnings], output_preview=t.output_preview,
            execution_time_s=t.execution_time_s, llm_calls=t.llm_calls, error=_error(t.error),
            checks=[c.model_copy(update={"message": redact(c.message)}) for c in t.checks],
        )
    return BuildDetail(
        **build_summary(record, result).model_dump(),
        constraints=result.constraints, plan=plan, tool_selection=tool, generation=gen, test=test,
        deployment=deployment_ref(record.id, result),
    )


def event_out(item: StreamEvent) -> BuildEventOut:
    return BuildEventOut(id=item.id, build_id=item.build_id, kind=item.kind, event=item.event,
                         stage=item.stage, message=item.message, attempt=item.attempt,
                         created_at=item.created_at)
