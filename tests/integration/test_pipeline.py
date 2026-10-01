import json

import pytest

from metaagent.ai.generator.generator import CodeGenerator
from metaagent.builds.service import BuildService
from metaagent.pipeline import run_build
from metaagent.schemas import CodeGenerationResult, Constraints
from tests.conftest import CODEGEN_MARKER, CRASHING_AGENT, GOOD_AGENT, FakeLLM, fenced

pytestmark = pytest.mark.subprocess

CHATBOT_PLAN = json.dumps({"agent_type": "chatbot", "tools": [],
                           "flow": ["parse_query", "generate_response", "return_output"]})
REQUEST = "Build a friendly chatbot for customer service"


def build(settings, llm, **kwargs):
    return run_build(REQUEST, Constraints(), "gemini-2.5-flash", llm=llm, settings=settings, **kwargs)


def test_successful_build_is_deployed(settings):
    events = []
    result = build(settings, FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)]), on_event=events.append)
    assert result.succeeded and result.deployed
    assert result.generation.method == "llm" and result.generation.corrections == 0
    assert result.test.passed and [c.name for c in result.test.checks] == [
        "syntax", "policy", "structure", "runtime", "output"]
    assert (settings.deployments_dir / result.build_id / "agent.py").exists()
    stages = [(e.stage, e.event) for e in events]
    assert ("planning", "started") in stages and ("deployment", "completed") in stages
    assert all(e.build_id == result.build_id for e in events)


def test_self_correction_feeds_errors_back(settings):
    llm = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(CRASHING_AGENT), fenced(GOOD_AGENT)])
    result = build(settings, llm)
    assert result.succeeded and result.generation.method == "llm"
    assert result.generation.corrections == 1
    assert [a.passed for a in result.generation.attempts] == [False, True]
    retry_prompt = [p for p in llm.prompts if CODEGEN_MARKER in p][1]
    assert "boom from generated code" in retry_prompt


def test_failed_tests_block_deployment(settings, monkeypatch):
    def broken_template(self, plan, selection, request):
        return CodeGenerationResult(code=CRASHING_AGENT, method="template", lines=3)

    monkeypatch.setattr(CodeGenerator, "generate_template", broken_template)
    llm = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(CRASHING_AGENT), fenced(CRASHING_AGENT)])
    result = build(settings, llm)

    assert not result.succeeded
    assert result.failed_stage == "testing"
    assert result.deployment is None and not result.deployed
    assert "boom from generated code" in result.error.message
    assert not settings.deployments_dir.exists()


def test_llm_generation_failure_falls_back_to_template(settings):
    llm = FakeLLM(plan=CHATBOT_PLAN, code=[RuntimeError("model unavailable")])
    result = build(settings, llm)
    assert result.succeeded and result.generation.method == "template"
    assert result.generation.attempts[0].error.startswith("Code generation LLM call failed")
    assert llm.count(CODEGEN_MARKER) == 1  # unreachable model: no pointless retry


def test_generation_failure_without_any_code_reports_generation_stage(settings, monkeypatch):
    def failing_template(self, plan, selection, request):
        return self._failed("TemplateError", "template exploded", 0.0)

    monkeypatch.setattr(CodeGenerator, "generate_template", failing_template)
    result = build(settings, FakeLLM(plan=CHATBOT_PLAN, code=[RuntimeError("down")]))
    assert not result.succeeded and result.failed_stage == "generation"
    assert result.deployment is None


def test_cache_hit_is_retested_and_failures_are_not_cached(settings):
    first = build(settings, FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)]))
    assert first.generation.method == "llm"

    second_llm = FakeLLM(plan=CHATBOT_PLAN)
    second = build(settings, second_llm)
    assert second.generation.method == "cache" and second.test.passed and second.deployed
    assert second_llm.count(CODEGEN_MARKER) == 0

    # Different constraints → different key → not served from cache.
    third = run_build(REQUEST, Constraints(budget="paid"), "gemini-2.5-flash", settings=settings,
                      llm=FakeLLM(plan=CHATBOT_PLAN, code=[RuntimeError("down")]))
    assert third.generation.method == "template"
    cache = json.loads(settings.cache_file.read_text())
    assert len(cache) == 1  # the template result was not cached


def test_unreachable_model_fails_at_planning_instead_of_building_a_chatbot(settings):
    llm = FakeLLM(plan=RuntimeError("429 RESOURCE_EXHAUSTED"))
    result = run_build("Build a PDF QA system for my documents", settings=settings, llm=llm)
    assert not result.succeeded and result.failed_stage == "planning"
    assert "quota" in result.error.message
    assert result.generation is None and result.deployment is None


def test_rag_agent_reporting_empty_store_passes(settings):
    rag_plan = json.dumps({"agent_type": "rag", "tools": ["retriever"], "flow": ["parse_query", "return_output"]})
    selection = json.dumps({"tools": {"retriever": {"implementation": "chromadb", "reasoning": "t"}}})
    empty_store_agent = (
        "class Agent:\n    def run(self, user_input: str) -> str:\n"
        "        return 'No documents loaded yet. Please upload files first.'\n"
    )
    result = run_build("Build a PDF QA system for my documents", settings=settings,
                       llm=FakeLLM(plan=rag_plan, selection=selection, code=[fenced(empty_store_agent)]))
    assert result.succeeded and result.generation.method == "llm"


def test_input_validation(settings):
    empty = run_build("   ", settings=settings, llm=FakeLLM())
    assert empty.failed_stage == "input" and not empty.deployed
    too_long = run_build("x" * (settings.max_request_chars + 1), settings=settings, llm=FakeLLM())
    assert too_long.failed_stage == "input"
    bad_model = run_build(REQUEST, model="gpt-4", settings=settings, llm=FakeLLM())
    assert bad_model.failed_stage == "input" and "not enabled" in bad_model.error.message


def test_build_service_persists_history(settings):
    from metaagent.builds import service as service_module

    service = BuildService(settings)
    original = service_module.run_build
    service_module.run_build = lambda *a, **k: original(*a, llm=FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)]), **k)
    try:
        result = service.start_build(REQUEST, Constraints(), "gemini-2.5-flash")
    finally:
        service_module.run_build = original
    assert service.store.get(result.build_id).build_id == result.build_id
    stats = service.stats()
    assert stats.total == 1 and stats.succeeded == 1
    live = service.run_deployed_agent(result, "ping")
    assert live.ok and live.output == "Echo: ping"
