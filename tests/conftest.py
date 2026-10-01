"""Shared fixtures. No test in this suite calls the real Gemini API."""

import pytest

from metaagent.core.config import Settings

PLANNER_MARKER = "expert AI system planner"
SELECTOR_MARKER = "tool selector for AI agents"
CODEGEN_MARKER = "expert Python developer"

GOOD_AGENT = '''
"""Echo agent used by tests."""


class Agent:
    def __init__(self):
        self.prefix = "Echo"

    def run(self, user_input: str) -> str:
        try:
            return f"{self.prefix}: {user_input}"
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    print(Agent().run("hi"))
'''

CRASHING_AGENT = '''
class Agent:
    def run(self, user_input: str) -> str:
        raise ValueError("boom from generated code")
'''


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Routes prompts to canned responses by recognising each stage's prompt.

    A response may be an Exception instance, which is raised instead.
    `code` is a list consumed one item per code-generation call.
    """

    def __init__(self, plan=None, selection=None, code=None):
        self.plan = plan
        self.selection = selection
        self.code = list(code or [])
        self.prompts = []

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        if PLANNER_MARKER in text:
            value = self.plan
        elif SELECTOR_MARKER in text:
            value = self.selection
        elif CODEGEN_MARKER in text:
            value = self.code.pop(0) if self.code else None
        else:
            value = None
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("FakeLLM has no response configured for this prompt")
        return FakeResponse(value)

    def count(self, marker: str) -> int:
        return sum(marker in p for p in self.prompts)


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", max_generation_attempts=2, test_run_timeout_s=60)
