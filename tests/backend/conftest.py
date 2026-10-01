"""Backend test fixtures. Real FastAPI app, real SQLite (temp file, migrated
with Alembic), real subprocess runner; fake LLM and fake embeddings, so no
test spends Gemini quota."""

import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import ApiKey, ApiSettings
from backend.app.main import create_app
from metaagent.rag.service import DocumentService
from tests.conftest import FakeLLM
from tests.integration.test_rag_service import EchoLLM, FakeEmbeddings

BUILDER_KEY = "builder-test-key-0123456789abcdef"
VIEWER_KEY = "viewer-test-key-0123456789abcdef"
BUILDER = {"Authorization": f"Bearer {BUILDER_KEY}"}
VIEWER = {"Authorization": f"Bearer {VIEWER_KEY}"}

CHATBOT_PLAN = json.dumps({"agent_type": "chatbot", "tools": [],
                           "flow": ["parse_query", "generate_response", "return_output"]})


class SwitchableLLM:
    """One LLM object for the whole app; each test sets `.current` to a FakeLLM."""

    def __init__(self):
        self.current = FakeLLM()

    def invoke(self, prompt):
        return self.current.invoke(prompt)


@pytest.fixture
def fake_llm():
    return SwitchableLLM()


@pytest.fixture
def api_settings(settings, tmp_path):
    return ApiSettings(
        engine=settings,
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        api_keys=(ApiKey("ci-builder", "builder", BUILDER_KEY), ApiKey("ci-viewer", "viewer", VIEWER_KEY)),
        sse_heartbeat_s=0.5,
    )


@pytest.fixture
def make_client(api_settings, fake_llm):
    clients = []

    def _make(**overrides) -> TestClient:
        s = replace(api_settings, **overrides)
        app = create_app(
            s, llm=fake_llm,
            document_service_factory=lambda collection: DocumentService(
                s.engine, embeddings=FakeEmbeddings(), llm=EchoLLM(), collection=collection),
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()  # run lifespan (migrations, container)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


# ── helpers ──

def start_build(client, request="Build a friendly chatbot for customer service", **body):
    response = client.post("/api/v1/builds", json={"request": request, **body}, headers=BUILDER)
    assert response.status_code == 202, response.text
    return response.json()["build_id"]


def wait_for_build(client, build_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/builds/{build_id}", headers=VIEWER).json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError(f"build {build_id} did not finish in {timeout}s")


def read_events(client, build_id, last_event_id=None, headers=VIEWER):
    """Consume an SSE stream until the server closes it; return parsed messages."""
    h = dict(headers)
    if last_event_id is not None:
        h["Last-Event-ID"] = str(last_event_id)
    events, current, comments = [], {}, 0
    with client.stream("GET", f"/api/v1/builds/{build_id}/events", headers=h) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith(":"):
                comments += 1
            elif line.startswith("id: "):
                current["id"] = int(line[4:])
            elif line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif line == "" and current:
                events.append(current)
                current = {}
    return events
