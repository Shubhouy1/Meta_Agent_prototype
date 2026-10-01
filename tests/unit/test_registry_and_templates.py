import ast
import itertools

import pytest

from metaagent.ai.generator.templates import render_template
from metaagent.core.config import Settings
from metaagent.execution.policy import check_code
from metaagent.schemas import VALID_TOOLS, ToolChoice
from metaagent.tools.registry import DEFAULT_IMPLEMENTATIONS, TOOL_REGISTRY, available_implementations


def test_registry_covers_exactly_the_planner_tools():
    assert set(TOOL_REGISTRY) == set(VALID_TOOLS)
    for tool, impl_id in DEFAULT_IMPLEMENTATIONS.items():
        assert impl_id in {i.id for i in available_implementations(tool)}


@pytest.mark.parametrize("tool,impl", [(t, i) for t, impls in TOOL_REGISTRY.items() for i in impls if i.available])
def test_every_available_snippet_defines_its_entrypoint(tool, impl):
    tree = ast.parse(impl.render_snippet(Settings()))
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert impl.entrypoint in defined
    assert check_code(impl.render_snippet(Settings())).passed


def test_unavailable_implementations_say_why():
    for impls in TOOL_REGISTRY.values():
        for impl in impls:
            if not impl.available:
                assert impl.unavailable_reason


def _choices(tools):
    return {t: ToolChoice(tool=t, implementation=DEFAULT_IMPLEMENTATIONS[t], name=t) for t in tools}


TOOL_COMBOS = [c for r in range(len(VALID_TOOLS) + 1) for c in itertools.combinations(VALID_TOOLS, r)]


@pytest.mark.parametrize("agent_type", ["chatbot", "rag"])
def test_templates_produce_valid_python(agent_type):
    code = render_template(agent_type, {}, "Build an agent")
    ast.parse(code)
    assert check_code(code).passed


@pytest.mark.parametrize("tools", TOOL_COMBOS)
def test_tool_agent_template_valid_for_every_tool_combination(tools):
    code = render_template("tool_agent", _choices(tools), "Build a tool agent")
    tree = ast.parse(code)
    assert check_code(code).passed
    assert any(isinstance(n, ast.ClassDef) and n.name == "Agent" for n in tree.body)
    if "memory" in tools:
        assert "self.memory = MemoryTool()" in code


def test_rag_template_uses_configured_embedding_model():
    code = render_template("rag", {}, "pdf qa", settings=Settings(embedding_model="my-embedding-model"))
    assert "EMBEDDING_MODEL = 'my-embedding-model'" in code
    assert ".persist()" not in code


@pytest.mark.parametrize("agent_type", ["chatbot", "rag", "tool_agent"])
def test_hostile_request_cannot_inject_code(agent_type):
    hostile = 'x"""\nimport os\nos.system("echo pwned")\n"""{e}{uploaded_file.name}$timestamp'
    code = render_template(agent_type, {}, hostile)
    tree = ast.parse(code)
    # The request survives only as a string constant, never as executable code.
    assert check_code(code).passed
    assigned = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "AGENT_REQUEST" for t in n.targets)]
    assert assigned and ast.literal_eval(assigned[0].value) == hostile
