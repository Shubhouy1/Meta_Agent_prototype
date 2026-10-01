"""The policy is a pre-filter, not a sandbox. These tests pin down what it
rejects, including the bypasses found in the audit."""

import pytest

from metaagent.execution.policy import check_code

AGENT_SUFFIX = "\nclass Agent:\n    def run(self, x):\n        return 'x'\n"


@pytest.mark.parametrize("snippet", [
    "import os\nos.system('dir')",
    "from os import system",
    "from os import *",
    "import subprocess",
    "import subprocess as sp",
    "from subprocess import run",
    "import importlib\nimportlib.import_module('sub' + 'process')",
    "import shutil",
    "import os\nprint(os.environ)",
    "open('.env').read()\neval('1')",
    "exec('print(1)')",
    "__import__('os')",
    "import google.generativeai as genai",
    "from google import generativeai",
    "import pickle",
])
def test_rejects_known_dangerous_constructs(snippet):
    result = check_code(snippet + AGENT_SUFFIX)
    assert not result.passed, snippet
    assert result.violations


def test_accepts_ordinary_agent_code():
    code = (
        "import os\nimport json\nfrom langchain_google_genai import ChatGoogleGenerativeAI\n"
        "CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')\n"
        "class Agent:\n    def __init__(self):\n        self.ok = os.path.isdir(CHROMA_DIR)\n"
        "    def run(self, user_input: str) -> str:\n        return json.dumps({'echo': user_input})\n"
    )
    assert check_code(code).passed


def test_syntax_error_is_reported():
    result = check_code("def broken(:\n")
    assert not result.passed and "Syntax error" in result.message
