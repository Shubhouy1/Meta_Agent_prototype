import json
import re
import threading

import pytest

from backend.app.db.build_store import SqlBuildStore
from backend.app.db.session import make_engine, run_migrations
from metaagent.ai.generator.generator import CodeGenerator
from metaagent.schemas import BuildResult, CodeGenerationResult
from tests.backend.conftest import (
    BUILDER, CHATBOT_PLAN, VIEWER, read_events, start_build, wait_for_build,
)
from tests.conftest import CRASHING_AGENT, GOOD_AGENT, FakeLLM, fenced

pytestmark = pytest.mark.subprocess

ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:/|/(home|Users|tmp)/")


@pytest.fixture
def broken_template(monkeypatch):
    monkeypatch.setattr(CodeGenerator, "generate_template",
                        lambda self, plan, sel, req: CodeGenerationResult(code=CRASHING_AGENT, method="template", lines=3))


# ── create / status ──

def test_create_build_returns_id_and_runs_to_deployment(client, fake_llm):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    response = client.post("/api/v1/builds", json={"request": "Build a friendly chatbot for customer service"},
                           headers=BUILDER)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["build_id"]
    assert body["events_url"] == f"/api/v1/builds/{body['build_id']}/events"

    build = wait_for_build(client, body["build_id"])
    assert build["status"] == "succeeded" and build["deployed"]
    assert build["owner"] == "ci-builder"
    assert build["agent_type"] == "chatbot"
    assert build["generation"]["method"] == "llm"
    assert build["test"]["passed"] and [c["name"] for c in build["test"]["checks"]] == [
        "syntax", "policy", "structure", "runtime", "output"]
    assert build["deployment"]["invoke_url"].endswith("/invoke")
    assert build["started_at"] and build["finished_at"]


def test_status_is_persisted_while_queued_and_running(client, fake_llm):
    gate = threading.Event()

    class SlowLLM(FakeLLM):
        def invoke(self, prompt):
            gate.wait(10)
            return super().invoke(prompt)

    fake_llm.current = SlowLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    build_id = start_build(client)
    status = client.get(f"/api/v1/builds/{build_id}", headers=VIEWER).json()["status"]
    assert status in ("queued", "running")
    gate.set()
    assert wait_for_build(client, build_id)["status"] == "succeeded"


@pytest.mark.parametrize("body,fragment", [
    ({"request": ""}, "Invalid request"),
    ({"request": "   "}, "Describe the agent"),
    ({"request": "Build a bot", "model": "gpt-4"}, "not enabled"),
    ({"request": "Build a bot", "constraints": {"budget": "unlimited"}}, "Invalid request"),
    ({"request": "Build a bot", "surprise": True}, "Invalid request"),
    ({}, "Invalid request"),
])
def test_invalid_requests_are_rejected(client, body, fragment):
    response = client.post("/api/v1/builds", json=body, headers=BUILDER)
    assert response.status_code == 422
    assert fragment in response.json()["detail"]


def test_request_longer_than_engine_limit_is_rejected(client):
    response = client.post("/api/v1/builds", json={"request": "x" * 2001}, headers=BUILDER)
    assert response.status_code == 422 and "too long" in response.json()["detail"]


def test_unknown_build(client):
    for path in ("/api/v1/builds/nope", "/api/v1/builds/nope/events"):
        response = client.get(path, headers=VIEWER)
        assert response.status_code == 404
        assert response.json() == {"detail": "Build not found"}


# ── history ──

def test_build_history_comes_from_the_database(client, fake_llm):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT), fenced(GOOD_AGENT)])
    first = start_build(client, "Build a friendly chatbot for customer service")
    wait_for_build(client, first)
    second = start_build(client, "Build a cheerful chatbot that greets visitors")
    wait_for_build(client, second)

    page = client.get("/api/v1/builds?limit=1", headers=VIEWER).json()
    assert [b["build_id"] for b in page["items"]] == [second] and page["limit"] == 1
    both = client.get("/api/v1/builds", headers=VIEWER).json()["items"]
    assert [b["build_id"] for b in both] == [second, first]
    assert client.get("/api/v1/builds?status=failed", headers=VIEWER).json()["items"] == []
    assert client.get("/api/v1/builds?limit=500", headers=VIEWER).status_code == 422
    assert client.get("/api/v1/builds?status=bogus", headers=VIEWER).status_code == 422


def test_history_survives_a_restart(make_client, fake_llm):
    first = make_client()
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    build_id = start_build(first)
    wait_for_build(first, build_id)
    second = make_client()  # new app, same database
    assert second.get(f"/api/v1/builds/{build_id}", headers=VIEWER).json()["status"] == "succeeded"


def test_interrupted_builds_are_marked_failed_on_startup(api_settings, make_client):
    run_migrations(api_settings.resolved_database_url)
    store = SqlBuildStore(make_engine(api_settings.resolved_database_url))
    store.create(BuildResult(build_id="stuck-build", request="x", status="running"), owner="ci-builder")
    client = make_client()
    build = client.get("/api/v1/builds/stuck-build", headers=VIEWER).json()
    assert build["status"] == "failed" and build["error"]["type"] == "Interrupted"
    events = read_events(client, "stuck-build")
    assert events[-1]["event"] == "status" and events[-1]["data"]["event"] == "failed"


def test_full_queue_returns_503(make_client, fake_llm):
    client = make_client(max_concurrent_builds=1, max_queued_builds=1)
    gate = threading.Event()

    class SlowLLM(FakeLLM):
        def invoke(self, prompt):
            gate.wait(10)
            return super().invoke(prompt)

    fake_llm.current = SlowLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    build_id = start_build(client)
    response = client.post("/api/v1/builds", json={"request": "Build another chatbot please"}, headers=BUILDER)
    assert response.status_code == 503 and response.headers["Retry-After"]
    gate.set()
    wait_for_build(client, build_id)


# ── events ──

def test_sse_streams_the_whole_successful_build(client, fake_llm):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    build_id = start_build(client)
    events = read_events(client, build_id)

    ids = [e["id"] for e in events]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    seen = [(e["data"]["stage"], e["data"]["event"]) for e in events if e["event"] == "stage"]
    for expected in [("planning", "started"), ("planning", "completed"), ("tool_selection", "started"),
                     ("generation", "started"), ("testing", "started"), ("deployment", "started"),
                     ("deployment", "completed")]:
        assert expected in seen
    statuses = [e["data"]["event"] for e in events if e["event"] == "status"]
    assert statuses == ["queued", "running", "succeeded"]
    assert events[-1]["event"] == "status"  # stream ends on the terminal event


def test_sse_reports_test_failure_self_correction_and_build_failure(client, fake_llm, broken_template):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(CRASHING_AGENT), fenced(CRASHING_AGENT)])
    build_id = start_build(client)
    events = read_events(client, build_id)
    kinds = [(e["data"]["stage"], e["data"]["event"]) for e in events]
    assert ("testing", "attempt_failed") in kinds
    assert ("generation", "self_correction") in kinds
    assert ("generation", "fallback") in kinds
    assert ("testing", "failed") in kinds
    assert ("deployment", "started") not in kinds
    last = events[-1]["data"]
    assert last["kind"] == "status" and last["event"] == "failed" and last["stage"] == "testing"

    build = client.get(f"/api/v1/builds/{build_id}", headers=VIEWER).json()
    assert build["status"] == "failed" and build["failed_stage"] == "testing" and not build["deployed"]
    assert build["deployment"] is None


def test_sse_resumes_from_last_event_id(client, fake_llm):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    build_id = start_build(client)
    events = read_events(client, build_id)
    middle = events[len(events) // 2]["id"]
    resumed = read_events(client, build_id, last_event_id=middle)
    assert [e["id"] for e in resumed] == [e["id"] for e in events if e["id"] > middle]
    assert read_events(client, build_id, last_event_id=events[-1]["id"]) == []


def test_api_responses_never_contain_absolute_paths(client, fake_llm, broken_template):
    leaky = CRASHING_AGENT.replace('"boom from generated code"', r'"boom at C:\\Users\\someone\\secret\\agent.py"')
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(leaky), fenced(leaky)])
    build_id = start_build(client)
    wait_for_build(client, build_id)
    build = client.get(f"/api/v1/builds/{build_id}", headers=VIEWER).json()
    # The generated source is agent content (it contains the planted string), not server state.
    build["generation"]["code"] = None
    body = json.dumps(build)
    events = json.dumps(read_events(client, build_id))
    for text in (body, events):
        assert "secret" not in text and "someone" not in text
        assert not ABSOLUTE_PATH.search(text)
    assert "<path>" in body
