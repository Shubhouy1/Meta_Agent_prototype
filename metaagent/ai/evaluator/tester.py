"""Stage 4: decide whether generated code is fit to deploy.

Checks, in order (later checks are skipped once one fails):
  1. syntax        - ast.parse
  2. policy        - static denylist (execution/policy.py; not a sandbox)
  3. structure     - class Agent with run(self, <input>) defined at module level
  4. runtime       - import, instantiate and call run() once, in an isolated
                     subprocess (execution/runner.py, TEST mode: no API key,
                     canned LLM replies, timeout)
  5. output        - run() returned a non-empty string that is not an error message
"""

import ast
import logging
import re
import time
from typing import List, Optional

from metaagent.execution.policy import check_code
from metaagent.execution.runner import AgentRunner, RunMode
from metaagent.schemas import StageError, StageStatus, TestCheck, TestResult

logger = logging.getLogger(__name__)

SAMPLE_INPUT = "Hello, this is a test"
_ERROR_OUTPUT = re.compile(r"^\s*(error\b|an error occurred|exception\b|traceback\b)", re.IGNORECASE)


def _structure_problem(tree: ast.Module) -> Optional[str]:
    agent = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent"), None)
    if agent is None:
        return "No module-level class Agent"
    run = next((n for n in agent.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "run"), None)
    if run is None:
        return "class Agent has no run() method"
    if isinstance(run, ast.AsyncFunctionDef):
        return "run() must be a regular method, not async"
    if len(run.args.args) < 2 and run.args.vararg is None:
        return "run() must accept the user input: run(self, user_input)"
    return None


class Stage4Tester:
    def __init__(self, runner: Optional[AgentRunner] = None):
        self.runner = runner or AgentRunner()

    def run_tests(self, code: str) -> TestResult:
        started = time.perf_counter()
        checks: List[TestCheck] = []
        warnings: List[str] = []

        def fail(check: str, message: str, detail: Optional[str] = None, **extra) -> TestResult:
            checks.append(TestCheck(name=check, passed=False, message=message))
            logger.info("Stage 4 failed at %s: %s", check, message)
            return TestResult(
                status=StageStatus.FAILED, passed=False, checks=checks, warnings=warnings,
                error=StageError(type=f"{check}_failed", message=message, detail=detail),
                duration_s=time.perf_counter() - started, **extra,
            )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return fail("syntax", f"Syntax error at line {e.lineno}: {e.msg}")
        checks.append(TestCheck(name="syntax", passed=True, message="Code parses"))

        policy = check_code(code)
        if not policy.passed:
            return fail("policy", policy.message)
        checks.append(TestCheck(name="policy", passed=True, message="No forbidden constructs found"))

        problem = _structure_problem(tree)
        if problem:
            return fail("structure", problem)
        checks.append(TestCheck(name="structure", passed=True, message="class Agent with run(self, input)"))

        lines = len(code.splitlines())
        if lines < 10:
            warnings.append(f"Agent code is very short ({lines} lines)")
        elif lines > 500:
            warnings.append(f"Agent code is very large ({lines} lines)")

        run = self.runner.run_code(code, SAMPLE_INPUT, mode=RunMode.TEST)
        extra = {"execution_time_s": run.run_time_s, "llm_calls": run.llm_calls}
        if not run.ok:
            message = run.error or "Agent run failed"
            if run.error_type and not run.timed_out:
                message = f"{run.error_type}: {message}"
            return fail("runtime", message, detail=run.traceback or run.stderr_tail or None, **extra)
        checks.append(TestCheck(name="runtime", passed=True,
                                message=f"Imported, instantiated and ran in {run.duration_s:.1f}s "
                                        f"({run.llm_calls} proxied LLM call(s))"))

        output = run.output
        if output is None:
            return fail("output", f"run() returned {run.output_type}, expected str", **extra)
        if not output.strip():
            return fail("output", "run() returned an empty string", **extra)
        if _ERROR_OUTPUT.match(output):
            return fail("output", f"run() returned an error message: {output[:300]}", **extra)
        checks.append(TestCheck(name="output", passed=True, message=f"Returned {len(output)} characters"))

        return TestResult(
            passed=True, checks=checks, warnings=warnings,
            output_preview=output[:500], duration_s=time.perf_counter() - started, **extra,
        )
