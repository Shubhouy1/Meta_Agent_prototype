import json

import pytest

from metaagent.schemas import Constraints, Plan
from metaagent.tools.registry import TOOL_REGISTRY
from metaagent.tools.selector import ToolSelector, score_implementation, simulate_options
from tests.conftest import FakeLLM, SELECTOR_MARKER


def selection(**tools):
    return json.dumps({"tools": {t: {"implementation": i, "reasoning": "test"} for t, i in tools.items()}})


def test_no_tools_needs_no_llm():
    llm = FakeLLM()
    result = ToolSelector(llm=llm).select_tools(Plan(agent_type="chatbot"), "chat")
    assert result.ok and result.selections == {} and result.selection_method == "none"
    assert llm.prompts == []


def test_valid_llm_choice_is_used():
    llm = FakeLLM(selection=selection(search="duckduckgo", calculator="safe_eval"))
    plan = Plan(agent_type="tool_agent", tools=["search", "calculator"])
    result = ToolSelector(llm=llm).select_tools(plan, "search and calculate")
    assert result.ok and result.selection_method == "llm"
    assert {t: c.implementation for t, c in result.selections.items()} == {
        "search": "duckduckgo", "calculator": "safe_eval"}
    assert llm.count(SELECTOR_MARKER) == 1


@pytest.mark.parametrize("bad_impl", ["made_up_engine", "pinecone"])
def test_unknown_or_unavailable_implementation_is_replaced(bad_impl):
    llm = FakeLLM(selection=selection(retriever=bad_impl))
    result = ToolSelector(llm=llm).select_tools(Plan(agent_type="rag", tools=["retriever"]), "pdf qa")
    assert result.selections["retriever"].implementation == "chromadb"
    assert any(bad_impl in w for w in result.warnings)


def test_tools_not_in_plan_are_ignored():
    llm = FakeLLM(selection=selection(search="duckduckgo", memory="buffer_memory"))
    result = ToolSelector(llm=llm).select_tools(Plan(agent_type="tool_agent", tools=["search"]), "search")
    assert list(result.selections) == ["search"]
    assert any("memory" in w for w in result.warnings)


def test_llm_failure_uses_simulation_recommendation():
    llm = FakeLLM(selection=RuntimeError("timeout"))
    result = ToolSelector(llm=llm).select_tools(Plan(agent_type="tool_agent", tools=["search"]), "search")
    assert result.ok and result.selection_method == "rule"
    assert result.selections["search"].implementation == result.simulations["search"].recommended_id


@pytest.mark.parametrize("constraints", [
    Constraints(budget=b, privacy=p, performance=f, setup_time=s)
    for b in ("free", "paid") for p in ("strict", "moderate", "none")
    for f in ("fast", "balanced") for s in ("quick", "any")
])
def test_score_equals_breakdown_for_every_constraint_combination(constraints):
    for tool in TOOL_REGISTRY:
        for option in simulate_options(tool, constraints).options:
            assert option.score == sum(option.score_breakdown.values())
            assert 0 <= option.score <= 100


def test_constraints_change_the_recommendation():
    tavily = next(i for i in TOOL_REGISTRY["search"] if i.id == "tavily")
    ddg = next(i for i in TOOL_REGISTRY["search"] if i.id == "duckduckgo")
    strict_free = Constraints(budget="free", privacy="strict")
    assert sum(score_implementation(ddg, strict_free).values()) > sum(score_implementation(tavily, strict_free).values())
    assert simulate_options("search", strict_free).recommended_id == "duckduckgo"
    # Paid budget + no privacy concern + fast: Tavily's higher performance wins.
    paid_fast = Constraints(budget="paid", privacy="none", performance="fast", setup_time="any")
    assert simulate_options("search", paid_fast).recommended_id == "tavily"


def test_unavailable_options_are_never_recommended():
    for tool in TOOL_REGISTRY:
        sim = simulate_options(tool, Constraints(budget="paid", performance="fast"))
        recommended = next(o for o in sim.options if o.id == sim.recommended_id)
        assert recommended.available
