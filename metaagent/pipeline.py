"""The MetaAgent engine entry point.

    result = run_build(request="Build a PDF QA system", constraints=Constraints(), model="gemini-2.5-flash")

Flow:  plan → select tools → [generate → test → (failed? feed errors back)] × N
       → template fallback if every LLM attempt failed → deploy only if tests passed

This module never imports Streamlit (or any UI/web framework). Progress is
reported through an optional `on_event` callback so any caller (Streamlit,
CLI, tests, a future HTTP API with SSE) can display it.
"""

import logging
import time
import uuid
from typing import Callable, List, Optional

from metaagent.ai import llm as llm_factory
from metaagent.ai.evaluator.tester import Stage4Tester
from metaagent.ai.generator import cache as code_cache
from metaagent.ai.generator.generator import CodeGenerator, finalize
from metaagent.ai.planner import generate_plan
from metaagent.core.config import Settings, get_settings
from metaagent.core.logging import build_context, log_event
from metaagent.deployments.deployer import Stage5Deployer
from metaagent.execution.runner import AgentRunner
from metaagent.schemas import (
    BuildEvent, BuildResult, BuildStage, CodeGenerationResult, Constraints, GenerationAttempt,
    StageError, StageStatus, TestResult,
)
from metaagent.tools.selector import ToolSelector

logger = logging.getLogger(__name__)

EventCallback = Callable[[BuildEvent], None]


def new_build_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class _Build:
    """Per-build bookkeeping: events, logging, timing."""

    def __init__(self, result: BuildResult, on_event: Optional[EventCallback]):
        self.result = result
        self.on_event = on_event
        self.started = time.perf_counter()

    def emit(self, stage: BuildStage, event: str, message: str = "", attempt: Optional[int] = None) -> None:
        level = logging.WARNING if event == "failed" else logging.INFO
        log_event(logger, f"stage_{event}", level=level, stage=stage, message=message)
        if self.on_event:
            try:
                self.on_event(BuildEvent(build_id=self.result.build_id, stage=stage, event=event,
                                         message=message, attempt=attempt))
            except Exception as e:  # a broken UI callback must never break the build
                logger.warning("on_event callback failed: %s", e)

    def fail(self, stage: BuildStage, error: StageError) -> BuildResult:
        self.result.status = "failed"
        self.result.failed_stage = stage
        self.result.error = error
        self.emit(stage, "failed", error.message)
        return self.finish()

    def finish(self) -> BuildResult:
        self.result.duration_s = time.perf_counter() - self.started
        log_event(logger, "build_finished", status=self.result.status,
                  failed_stage=self.result.failed_stage or "-",
                  deployed=self.result.deployed, duration_s=f"{self.result.duration_s:.2f}")
        return self.result


def run_build(request: str, constraints: Optional[Constraints] = None, model: Optional[str] = None,
              *, build_id: Optional[str] = None, on_event: Optional[EventCallback] = None,
              llm=None, runner: Optional[AgentRunner] = None, settings: Optional[Settings] = None,
              use_cache: bool = True) -> BuildResult:
    """Run the full pipeline. Never raises for pipeline failures; inspect the result.

    `llm` and `runner` exist for tests and alternative callers; by default the
    configured Gemini client and the subprocess runner are used.
    """
    settings = settings or get_settings()
    constraints = constraints or Constraints()
    build_id = build_id or new_build_id()
    request = (request or "").strip()
    result = BuildResult(build_id=build_id, request=request, model=model, constraints=constraints)

    with build_context(build_id):
        build = _Build(result, on_event)
        log_event(logger, "build_started", model=model or settings.default_model,
                  constraints=constraints.model_dump(), request_chars=len(request))
        try:
            return _run_stages(build, request, constraints, model, llm, runner, settings, use_cache)
        except Exception as e:  # last line of defence: report, don't crash the caller
            logger.exception("Unexpected pipeline error")
            stage = result.failed_stage or "generation"
            return build.fail(stage, StageError(type=type(e).__name__, message=f"Unexpected error: {e}"))


def _run_stages(build: _Build, request: str, constraints: Constraints, model: Optional[str],
                llm, runner: Optional[AgentRunner], settings: Settings, use_cache: bool) -> BuildResult:
    result = build.result

    # ── Input ──
    if not request:
        return build.fail("input", StageError(type="ValidationError", message="Describe the agent you want to build."))
    if len(request) > settings.max_request_chars:
        return build.fail("input", StageError(
            type="ValidationError",
            message=f"Request is too long ({len(request)} characters, max {settings.max_request_chars})."))
    try:
        model = llm_factory.resolve_model(model, settings)
    except llm_factory.ModelNotAllowedError as e:
        return build.fail("input", StageError(type="ValidationError", message=str(e)))
    result.model = model

    # ── Stage 1: planning ──
    build.emit("planning", "started")
    plan_result = generate_plan(request, model=model, llm=llm)
    result.plan = plan_result
    if not plan_result.ok:
        return build.fail("planning", plan_result.error)
    plan = plan_result.plan
    build.emit("planning", "completed", f"agent_type={plan.agent_type} tools={plan.tools}")

    # ── Stage 2: tool selection ──
    build.emit("tool_selection", "started")
    selection = ToolSelector(llm=llm, model=model).select_tools(plan, request, constraints)
    result.tool_selection = selection
    if not selection.ok:
        return build.fail("tool_selection", selection.error)
    build.emit("tool_selection", "completed",
               ", ".join(f"{t}={c.implementation}" for t, c in selection.selections.items()) or "no tools")

    # ── Stages 3+4: generate ⇄ test (self-correction) ──
    tester = Stage4Tester(runner or AgentRunner(settings))
    generator = CodeGenerator(llm=llm, model=model, settings=settings)
    cache = code_cache.CodeCache(settings.cache_file) if use_cache else None
    cache_key = code_cache.make_key(
        request, plan.agent_type, {t: c.implementation for t, c in selection.selections.items()},
        constraints, model)

    attempts: List[GenerationAttempt] = []
    errors: List[str] = []
    generation: Optional[CodeGenerationResult] = None
    test: Optional[TestResult] = None
    gen_started = time.perf_counter()

    build.emit("generation", "started")

    # Cache hit: reuse only if it still passes the same tests.
    cached = cache.get(cache_key) if cache else None
    if cached:
        build.emit("generation", "cache_hit", "found a cached generation; re-testing it")
        build.emit("testing", "started", "re-testing cached code")
        candidate = finalize(cached["code"], "cache", cached.get("model"), gen_started)
        cached_test = tester.run_tests(candidate.code)
        attempts.append(GenerationAttempt(attempt=0, method="cache", passed=cached_test.passed,
                                          error=None if cached_test.passed else cached_test.error.message))
        if cached_test.passed:
            generation, test = candidate, cached_test
        else:
            cache.evict(cache_key)
            build.emit("testing", "info", "cached code no longer passes; evicted")

    for attempt in range(1, settings.max_generation_attempts + 1):
        if test is not None and test.passed:
            break
        if errors:
            build.emit("generation", "self_correction",
                       f"LLM attempt {attempt}/{settings.max_generation_attempts} with feedback from "
                       f"{len(errors)} failed attempt(s)", attempt=attempt)
        else:
            build.emit("generation", "info", f"LLM attempt {attempt}/{settings.max_generation_attempts}",
                       attempt=attempt)
        candidate = generator.generate_llm(plan, selection, request, previous_errors=errors or None)
        if not candidate.ok:
            errors.append(candidate.error.message)
            attempts.append(GenerationAttempt(attempt=attempt, method="llm", passed=False,
                                              error=candidate.error.message))
            build.emit("generation", "attempt_failed", candidate.error.message, attempt=attempt)
            if candidate.error.type == "LLMError":
                break  # the model is unreachable; retrying the prompt won't help
            continue
        build.emit("testing", "started", f"testing LLM attempt {attempt}", attempt=attempt)
        attempt_test = tester.run_tests(candidate.code)
        attempts.append(GenerationAttempt(attempt=attempt, method="llm", passed=attempt_test.passed,
                                          error=None if attempt_test.passed else attempt_test.error.message))
        generation, test = candidate, attempt_test
        if attempt_test.passed:
            if cache:
                cache.put(cache_key, candidate.code, model)
            break
        errors.append(attempt_test.error.message)
        build.emit("testing", "attempt_failed", f"attempt {attempt} failed: {attempt_test.error.message}",
                   attempt=attempt)

    if test is None or not test.passed:
        build.emit("generation", "fallback", "LLM attempts exhausted; using template fallback")
        candidate = generator.generate_template(plan, selection, request)
        if candidate.ok:
            build.emit("testing", "started", "testing template")
            template_test = tester.run_tests(candidate.code)
            attempts.append(GenerationAttempt(attempt=len(attempts) + 1, method="template",
                                              passed=template_test.passed,
                                              error=None if template_test.passed else template_test.error.message))
            generation, test = candidate, template_test
        else:
            attempts.append(GenerationAttempt(attempt=len(attempts) + 1, method="template",
                                              passed=False, error=candidate.error.message))
            if generation is None:
                generation = candidate

    llm_attempts = sum(1 for a in attempts if a.method == "llm")
    generation.attempts = attempts
    # Self-corrections = LLM retries that were given the previous errors as feedback.
    generation.corrections = max(0, llm_attempts - 1)
    generation.duration_s = time.perf_counter() - gen_started
    result.generation = generation
    result.test = test

    if not generation.code or (not generation.ok and test is None):
        return build.fail("generation", generation.error or StageError(
            type="GenerationFailed", message="No code could be generated"))
    build.emit("generation", "completed", f"method={generation.method} lines={generation.lines}")

    if test is None or not test.passed:
        result.deployment = None
        build.emit("deployment", "skipped", "tests failed")
        return build.fail("testing", test.error if test else StageError(
            type="TestFailed", message="Generated code was never tested"))
    build.emit("testing", "completed", "all checks passed")

    # ── Stage 5: deployment (only reachable when tests passed) ──
    build.emit("deployment", "started")
    deployment = Stage5Deployer(settings).deploy(result.build_id, generation.code, plan.agent_type, test)
    result.deployment = deployment
    if not deployment.ok:
        return build.fail("deployment", deployment.error)
    build.emit("deployment", "completed", f"deployment {deployment.deployment_id} is ready")

    result.status = "succeeded"
    return build.finish()
