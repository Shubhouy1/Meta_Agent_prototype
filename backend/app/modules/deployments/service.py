"""Deployments application service.

    route → DeploymentsService → metaagent BuildService.run_deployed_agent → AgentRunner (subprocess)

A deployment is the published artifact of a build that passed testing; its
id is currently the build id. Invocation reuses the engine's runner: the
agent runs in a separate process without API keys, its model calls are made
by this server, and a timeout kills it. That runner is NOT a hardened
sandbox (no filesystem, network or resource isolation), which is why
invocation requires an authenticated builder key.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from backend.app.core.errors import Conflict, NotFound, redact
from backend.app.core.rate_limit import ConcurrencyGate
from backend.app.db.build_store import SqlBuildStore
from backend.app.db.models import BuildRecord
from metaagent.ai.llm import describe_error
from metaagent.builds.service import BuildService
from metaagent.schemas import BuildResult

logger = logging.getLogger(__name__)


@dataclass
class Invocation:
    ok: bool
    output: Optional[str]
    error: Optional[str]
    timed_out: bool
    duration_s: float
    model_calls: int
    warnings: List[str] = field(default_factory=list)


class DeploymentsService:
    def __init__(self, engine: BuildService, store: SqlBuildStore, gate: ConcurrencyGate):
        self.engine = engine
        self.store = store
        self.gate = gate

    def get(self, deployment_id: str) -> Tuple[BuildRecord, BuildResult]:
        record = self.store.get_record(deployment_id)
        if record is None or not record.deployed:
            raise NotFound("Deployment not found")
        return record, BuildResult.model_validate_json(record.result_json)

    def invoke(self, deployment_id: str, user_input: str) -> Invocation:
        record = self.store.get_record(deployment_id)
        if record is None:
            raise NotFound("Deployment not found")
        result = BuildResult.model_validate_json(record.result_json)
        if not result.deployed:
            reason = f" (build {record.status}" + (f" at {record.failed_stage})" if record.failed_stage else ")")
            raise Conflict(f"Build {deployment_id} was not deployed{reason}; only agents that passed "
                           "testing can be invoked")

        with self.gate.slot():
            try:
                run = self.engine.run_deployed_agent(result, user_input)
            except FileNotFoundError:
                logger.error("Deployment %s artifact is missing on disk", deployment_id)
                raise NotFound("Deployment artifact is missing on the server")

        error = None
        if not run.ok:
            error = run.error if run.timed_out else redact(f"{run.error_type or 'Error'}: {run.error or ''}".strip())
        return Invocation(
            ok=run.ok, output=run.output if run.ok else None, error=error, timed_out=run.timed_out,
            duration_s=round(run.duration_s, 3), model_calls=run.llm_calls,
            # proxy errors are already short summaries; keep them unique and path-free
            warnings=[redact(describe_error(w)) for w in dict.fromkeys(run.proxy_errors)],
        )
