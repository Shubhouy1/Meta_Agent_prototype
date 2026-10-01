"""Every fallback template must pass Stage 4 for real (in the subprocess runner,
with the proxied test LLM). This is the check that would have caught the
original RAG template crash."""

import pytest

from metaagent.ai.evaluator.tester import Stage4Tester
from metaagent.ai.generator.templates import render_template
from metaagent.execution.runner import AgentRunner
from metaagent.schemas import ToolChoice
from metaagent.tools.registry import DEFAULT_IMPLEMENTATIONS

pytestmark = pytest.mark.subprocess


def _choices(*tools):
    return {t: ToolChoice(tool=t, implementation=DEFAULT_IMPLEMENTATIONS[t], name=t) for t in tools}


@pytest.mark.parametrize("agent_type,selections", [
    ("chatbot", {}),
    ("rag", {}),
    ("tool_agent", _choices("calculator", "memory")),
    ("tool_agent", _choices("search", "calculator", "retriever", "memory")),
])
def test_template_passes_stage_4(settings, agent_type, selections):
    code = render_template(agent_type, selections, "Build an agent", settings=settings)
    result = Stage4Tester(AgentRunner(settings)).run_tests(code)
    assert result.passed, (result.error, [c.model_dump() for c in result.checks])
