"""Typed contracts between pipeline stages.

Every stage result shares the same envelope (status, error, duration) so the
success and failure paths have the same shape. These models are the public
contract of the engine; a future HTTP API can serialise them directly.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

AgentType = Literal["chatbot", "rag", "tool_agent"]
ToolName = Literal["search", "calculator", "retriever", "memory"]

VALID_AGENT_TYPES = ("chatbot", "rag", "tool_agent")
VALID_TOOLS = ("search", "calculator", "retriever", "memory")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StageStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageError(BaseModel):
    type: str
    message: str
    detail: Optional[str] = None


class StageResult(BaseModel):
    status: StageStatus = StageStatus.SUCCEEDED
    error: Optional[StageError] = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == StageStatus.SUCCEEDED


# ── Inputs ──────────────────────────────────────────────

class Constraints(BaseModel):
    budget: Literal["free", "paid"] = "free"
    privacy: Literal["strict", "moderate", "none"] = "moderate"
    performance: Literal["fast", "balanced"] = "balanced"
    setup_time: Literal["quick", "any"] = "quick"


# ── Stage 1: planning ──────────────────────────────────

class Plan(BaseModel):
    agent_type: AgentType = "chatbot"
    tools: List[ToolName] = Field(default_factory=list)
    flow: List[str] = Field(default_factory=lambda: ["parse_query", "generate_response", "return_output"])
    confidence: float = 0.5
    version: str = "v1"


class PlanResult(StageResult):
    plan: Plan = Field(default_factory=Plan)
    fallback_used: bool = False
    warnings: List[str] = Field(default_factory=list)


# ── Stage 2: tool selection ────────────────────────────

class ToolOption(BaseModel):
    id: str
    name: str
    cost: str
    api_key_required: bool
    available: bool
    performance: str
    setup_time: str
    score: int
    score_breakdown: Dict[str, int] = Field(default_factory=dict)


class ToolSimulation(BaseModel):
    tool: str
    options: List[ToolOption] = Field(default_factory=list)
    recommended_id: Optional[str] = None
    recommended_name: Optional[str] = None
    reasoning: str = ""


class ToolChoice(BaseModel):
    tool: ToolName
    implementation: str
    name: str
    reasoning: str = ""
    was_recommended: bool = False
    api_key_required: bool = False
    cost: str = "Free"


class ToolSelectionResult(StageResult):
    selections: Dict[str, ToolChoice] = Field(default_factory=dict)
    simulations: Dict[str, ToolSimulation] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)
    selection_method: Literal["none", "llm", "rule"] = "none"
    warnings: List[str] = Field(default_factory=list)


# ── Stage 3: code generation ───────────────────────────

GenerationMethod = Literal["llm", "cache", "template"]


class GenerationAttempt(BaseModel):
    attempt: int
    method: GenerationMethod
    passed: bool
    error: Optional[str] = None


class CodeGenerationResult(StageResult):
    code: Optional[str] = None
    method: Optional[GenerationMethod] = None
    model: Optional[str] = None
    lines: int = 0
    quality_score: float = 0.0
    corrections: int = 0
    attempts: List[GenerationAttempt] = Field(default_factory=list)


# ── Stage 4: testing ───────────────────────────────────

class TestCheck(BaseModel):
    __test__ = False  # keep pytest from collecting this model

    name: str
    passed: bool
    message: str = ""


class TestResult(StageResult):
    __test__ = False

    passed: bool = False
    checks: List[TestCheck] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    output_preview: Optional[str] = None
    execution_time_s: Optional[float] = None
    llm_calls: int = 0


# ── Stage 5: deployment ────────────────────────────────

class DeploymentResult(StageResult):
    deployment_id: Optional[str] = None
    agent_type: Optional[AgentType] = None
    deployed_file: Optional[str] = None
    run_command: Optional[str] = None


# ── Whole build ────────────────────────────────────────

BuildStage = Literal["input", "planning", "tool_selection", "generation", "testing", "deployment"]


class BuildResult(BaseModel):
    build_id: str
    request: str
    model: Optional[str] = None
    constraints: Constraints = Field(default_factory=Constraints)
    status: Literal["succeeded", "failed"] = "failed"
    failed_stage: Optional[BuildStage] = None
    error: Optional[StageError] = None
    plan: Optional[PlanResult] = None
    tool_selection: Optional[ToolSelectionResult] = None
    generation: Optional[CodeGenerationResult] = None
    test: Optional[TestResult] = None
    deployment: Optional[DeploymentResult] = None
    started_at: datetime = Field(default_factory=utcnow)
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @property
    def deployed(self) -> bool:
        return self.deployment is not None and self.deployment.ok


class BuildEvent(BaseModel):
    """Progress notification emitted by the pipeline (UI-agnostic)."""
    build_id: str
    stage: BuildStage
    event: Literal["started", "completed", "failed", "skipped", "info"]
    message: str = ""
