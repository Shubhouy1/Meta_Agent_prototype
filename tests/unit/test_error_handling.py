"""Provider failures must surface as short, accurate messages, never as a
silently different build or a raw JSON dump."""

import pytest

from metaagent.ai.evaluator.tester import _ERROR_OUTPUT
from metaagent.ai.llm import describe_error

RAW_429 = ("Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
           "{'error': {'code': 429, 'message': 'You exceeded your current quota'}}")


@pytest.mark.parametrize("raw,expected", [
    (RAW_429, "quota or rate limit exceeded"),
    ("400 API key not valid. Please pass a valid API key. [reason: API_KEY_INVALID]", "API key was rejected"),
    ("403 PERMISSION_DENIED", "denied the request"),
    ("504 DEADLINE_EXCEEDED", "timed out"),
])
def test_describe_error_summarises_provider_errors(raw, expected):
    summary = describe_error(RuntimeError(raw))
    assert expected in summary
    assert "{'error'" not in summary


def test_describe_error_keeps_unknown_errors_short():
    assert len(describe_error(RuntimeError("x" * 5000))) <= 300


@pytest.mark.parametrize("output,is_error", [
    ("Error: model failed", True),
    ("An error occurred while processing", True),
    ("Traceback (most recent call last): ...", True),
    ("No documents loaded yet. Please upload files first.", False),
    ("Hello! How can I help you?", False),
    ("Errors in your essay: none found", False),
])
def test_output_check_distinguishes_errors_from_normal_messages(output, is_error):
    assert bool(_ERROR_OUTPUT.match(output)) is is_error
