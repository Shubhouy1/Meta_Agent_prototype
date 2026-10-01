import pytest

from metaagent.execution.runner import TEST_LLM_REPLY, AgentRunner, RunMode
from tests.conftest import GOOD_AGENT

pytestmark = pytest.mark.subprocess


@pytest.fixture
def runner(settings):
    return AgentRunner(settings)


def test_good_agent_runs(runner):
    result = runner.run_code(GOOD_AGENT, "hello")
    assert result.ok and result.output == "Echo: hello"


def test_api_keys_are_not_passed_to_generated_code(runner, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "parent-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "parent-secret-2")
    code = '''
import os
from dotenv import load_dotenv
load_dotenv()
class Agent:
    def run(self, x):
        return f"{os.getenv('GOOGLE_API_KEY')}|{os.getenv('TAVILY_API_KEY')}"
'''
    result = runner.run_code(code, "x")
    assert result.ok
    assert "parent-secret" not in result.output
    assert result.output == "None|None"


def test_prints_do_not_break_the_protocol(runner):
    code = 'class Agent:\n    def run(self, x):\n        print("{\\"type\\": \\"result\\"}")\n        return "fine"\n'
    assert runner.run_code(code, "x").output == "fine"


def test_infinite_loop_is_killed(runner):
    result = runner.run_code("class Agent:\n    def run(self, x):\n        while True:\n            pass\n",
                             "x", timeout_s=3)
    assert not result.ok and result.timed_out
    assert result.duration_s < 15


def test_exceptions_and_sys_exit_are_reported(runner):
    crash = runner.run_code("class Agent:\n    def run(self, x):\n        raise ValueError('boom')\n", "x")
    assert not crash.ok and crash.error_type == "ValueError" and "boom" in crash.error
    leave = runner.run_code("import sys\nclass Agent:\n    def run(self, x):\n        sys.exit(2)\n", "x")
    assert not leave.ok and leave.error_type == "SystemExit"


def test_input_does_not_block(runner):
    result = runner.run_code("class Agent:\n    def run(self, x):\n        return input()\n", "x", timeout_s=20)
    assert not result.ok and result.error_type == "EOFError"


def test_llm_calls_are_proxied_in_test_mode(runner):
    code = '''
from langchain_google_genai import ChatGoogleGenerativeAI
class Agent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    def run(self, x):
        return self.llm.invoke(x).content
'''
    result = runner.run_code(code, "hi", mode=RunMode.TEST)
    assert result.ok and result.output == TEST_LLM_REPLY and result.llm_calls == 1


def test_live_mode_uses_parent_side_model(settings):
    seen = {}

    def fake_chat(messages, model):
        seen["messages"], seen["model"] = messages, model
        return "live answer"

    runner = AgentRunner(settings, live_chat=fake_chat)
    code = '''
from langchain_google_genai import ChatGoogleGenerativeAI
class Agent:
    def run(self, x):
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash").invoke(x).content
'''
    result = runner.run_code(code, "question?", mode=RunMode.LIVE)
    assert result.ok and result.output == "live answer"
    assert seen["messages"][-1]["content"] == "question?"


def test_failed_proxy_calls_are_recorded_and_summarised(settings):
    def quota_exceeded(messages, model):
        raise RuntimeError("429 RESOURCE_EXHAUSTED {'error': {'code': 429}}")

    runner = AgentRunner(settings, live_chat=quota_exceeded)
    code = '''
from langchain_google_genai import ChatGoogleGenerativeAI
class Agent:
    def run(self, x):
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash").invoke(x).content
        except Exception as e:
            return f"Error: {e}"
'''
    result = runner.run_code(code, "hi", mode=RunMode.LIVE)
    assert result.ok  # the agent swallowed the error...
    assert result.proxy_errors and "quota" in result.proxy_errors[0]  # ...but the runner saw it
    assert "{'error'" not in result.output  # and the agent only received the summary


def test_llm_call_limit(settings):
    runner = AgentRunner(settings.__class__(data_dir=settings.data_dir, max_llm_calls_per_run=2))
    code = '''
from langchain_google_genai import ChatGoogleGenerativeAI
class Agent:
    def run(self, x):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        for _ in range(10):
            llm.invoke(x)
        return "done"
'''
    result = runner.run_code(code, "x")
    assert not result.ok and "limit" in result.error
