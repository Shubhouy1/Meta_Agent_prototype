import ast
import re
from pathlib import Path

import pytest

from metaagent.ai.generator.cache import CodeCache, make_key
from metaagent.deployments.deployer import DeploymentRefused, Stage5Deployer
from metaagent.schemas import Constraints, TestCheck, TestResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── cache ──

BASE = dict(user_request="Build a bot", agent_type="chatbot", implementations={},
            constraints=Constraints(), model="gemini-2.5-flash")


@pytest.mark.parametrize("change", [
    {"constraints": Constraints(budget="paid")},
    {"model": "gemini-2.5-pro"},
    {"agent_type": "rag"},
    {"implementations": {"search": "tavily"}},
    {"user_request": "Build a different bot"},
])
def test_cache_key_changes_with_every_input(change):
    assert make_key(**BASE) != make_key(**{**BASE, **change})


def test_cache_key_includes_prompt_version(monkeypatch):
    before = make_key(**BASE)
    monkeypatch.setattr("metaagent.ai.generator.cache.PROMPT_VERSION", "codegen-999")
    assert make_key(**BASE) != before


def test_cache_put_get_evict(tmp_path):
    cache = CodeCache(tmp_path / "cache.json")
    key = make_key(**BASE)
    assert cache.get(key) is None
    cache.put(key, "class Agent: ...", "gemini-2.5-flash")
    assert cache.get(key)["code"] == "class Agent: ..."
    cache.evict(key)
    assert cache.get(key) is None


def test_corrupt_cache_file_is_ignored(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json", encoding="utf-8")
    assert CodeCache(path).get("anything") is None


# ── deployer ──

def test_deployer_refuses_untested_or_failed_code(settings):
    deployer = Stage5Deployer(settings)
    with pytest.raises(DeploymentRefused):
        deployer.deploy("b1", "class Agent: ...", "chatbot", None)
    with pytest.raises(DeploymentRefused):
        deployer.deploy("b2", "class Agent: ...", "chatbot", TestResult(passed=False))
    assert not settings.deployments_dir.exists()


def test_deployments_are_versioned_per_build(settings):
    deployer = Stage5Deployer(settings)
    passed = TestResult(passed=True, checks=[TestCheck(name="syntax", passed=True)])
    first = deployer.deploy("build-1", "# one", "chatbot", passed)
    second = deployer.deploy("build-2", "# two", "chatbot", passed)
    assert first.deployed_file != second.deployed_file
    assert Path(first.deployed_file).read_text() == "# one"
    assert (settings.deployments_dir / "build-1" / "manifest.json").exists()


# ── execution paths ──

_ALLOWED = {Path("metaagent/execution/runner.py"), Path("metaagent/execution/_bootstrap.py")}
_EXEC_ATTRS = {"exec_module", "spec_from_file_location", "system", "popen", "spawnv", "execv"}
_EXEC_MODULES = {"subprocess", "multiprocessing"}


def _execution_sites(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _EXEC_MODULES:
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _EXEC_MODULES:
            yield node.lineno, f"from {node.module} import"
        elif isinstance(node, ast.Attribute) and node.attr in _EXEC_ATTRS:
            yield node.lineno, f".{node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            yield node.lineno, f"{node.func.id}()"


def test_generated_code_runs_only_through_the_runner():
    """Only execution/runner.py (and the child bootstrap) may execute code."""
    offenders = []
    files = list((PROJECT_ROOT / "metaagent").rglob("*.py")) + [PROJECT_ROOT / "app.py"]
    for path in files:
        rel = path.relative_to(PROJECT_ROOT)
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{rel}:{line}: {what}" for line, what in _execution_sites(tree)]
    assert offenders == []


def test_engine_does_not_import_streamlit():
    for path in (PROJECT_ROOT / "metaagent").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+streamlit", text, re.MULTILINE), path
