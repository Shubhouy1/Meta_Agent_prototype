from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.errors import redact
from backend.app.db.models import BuildRecord
from backend.app.modules.builds.api.schemas import API_PREFIX, as_utc
from backend.app.modules.deployments.service import Invocation
from metaagent.schemas import BuildResult, TestCheck


class InvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid",
                              json_schema_extra={"examples": [{"input": "What is 12 * 7?"}]})

    input: str = Field(min_length=1, max_length=4000, description="Message passed to the agent's run() method")


class DeploymentOut(BaseModel):
    deployment_id: str
    build_id: str
    status: str = Field(description="Always 'ready' for deployments that exist")
    agent_type: Optional[str]
    request: str
    model: Optional[str]
    deployed_at: Optional[datetime]
    checks: List[TestCheck] = Field(description="The Stage 4 checks this agent passed")
    build_url: str
    invoke_url: str


class InvokeResponse(BaseModel):
    deployment_id: str
    ok: bool = Field(description="False if the agent raised, timed out or returned a non-string")
    output: Optional[str]
    error: Optional[str]
    timed_out: bool
    duration_s: float
    model_calls: int = Field(description="Model calls made on the agent's behalf by the server")
    warnings: List[str] = Field(description="e.g. model calls that failed (quota) even if the agent "
                                            "returned its own text")


def deployment_out(record: BuildRecord, result: BuildResult) -> DeploymentOut:
    dep_id = result.deployment.deployment_id or record.id
    checks = [c.model_copy(update={"message": redact(c.message)}) for c in (result.test.checks if result.test else [])]
    return DeploymentOut(
        deployment_id=dep_id, build_id=record.id, status="ready", agent_type=result.deployment.agent_type,
        request=record.request, model=record.model, deployed_at=as_utc(record.finished_at), checks=checks,
        build_url=f"{API_PREFIX}/builds/{record.id}", invoke_url=f"{API_PREFIX}/deployments/{dep_id}/invoke",
    )


def invoke_response(deployment_id: str, run: Invocation) -> InvokeResponse:
    return InvokeResponse(deployment_id=deployment_id, ok=run.ok, output=run.output, error=run.error,
                          timed_out=run.timed_out, duration_s=run.duration_s, model_calls=run.model_calls,
                          warnings=run.warnings)
