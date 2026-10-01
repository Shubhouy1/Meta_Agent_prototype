import ast
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.core.config import parse_api_keys
from backend.app.core.dependencies import get_builds_service
from tests.backend.conftest import BUILDER, BUILDER_KEY, VIEWER

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTECTED = [
    ("post", "/api/v1/builds", {"json": {"request": "Build a bot please"}}),
    ("get", "/api/v1/builds", {}),
    ("get", "/api/v1/builds/x", {}),
    ("get", "/api/v1/builds/x/events", {}),
    ("get", "/api/v1/deployments/x", {}),
    ("post", "/api/v1/deployments/x/invoke", {"json": {"input": "hi"}}),
    ("get", "/api/v1/rag/documents", {}),
    ("post", "/api/v1/rag/query", {"json": {"question": "hi"}}),
    ("delete", "/api/v1/rag/documents/x", {}),
]


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_unauthenticated_requests_are_rejected(client, method, path, kwargs):
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing API key"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer wrong-key-wrong-key-wrong-key"},
    {"Authorization": f"Basic {BUILDER_KEY}"},
    {"Authorization": BUILDER_KEY},
])
def test_invalid_credentials_are_rejected(client, headers):
    assert client.get("/api/v1/builds", headers=headers).status_code == 401


def test_viewer_cannot_start_builds(client):
    response = client.post("/api/v1/builds", json={"request": "Build a bot please"}, headers=VIEWER)
    assert response.status_code == 403
    assert "may not perform this action" in response.json()["detail"]


def test_server_without_keys_fails_closed(make_client):
    client = make_client(api_keys=())
    response = client.get("/api/v1/builds", headers=BUILDER)
    assert response.status_code == 401 and "not configured" in response.json()["detail"]


def test_public_endpoints(client):
    assert client.get("/health").json() == {"status": "ok", "version": "0.2.0"}
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "HTTPBearer" in spec["components"]["securitySchemes"]
    assert {"builds", "deployments", "rag"} <= {t["name"] for t in spec["tags"]}


def test_docs_can_be_disabled(make_client):
    client = make_client(docs_enabled=False)
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_build_rate_limit(make_client, fake_llm):
    from tests.backend.conftest import CHATBOT_PLAN, wait_for_build
    from tests.conftest import GOOD_AGENT, FakeLLM, fenced

    client = make_client(builds_per_hour=1)
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(GOOD_AGENT)])
    first = client.post("/api/v1/builds", json={"request": "Build a friendly chatbot please"}, headers=BUILDER)
    assert first.status_code == 202
    second = client.post("/api/v1/builds", json={"request": "Build another chatbot please"}, headers=BUILDER)
    assert second.status_code == 429 and "Rate limit" in second.json()["detail"]
    assert int(second.headers["Retry-After"]) > 0
    wait_for_build(client, first.json()["build_id"])


def test_request_body_size_limit(make_client):
    client = make_client(max_body_bytes=1000)
    big = {"request": "x" * 5000}
    response = client.post("/api/v1/builds", json=big, headers=BUILDER)
    assert response.status_code == 413 and "too large" in response.json()["detail"]
    # Chunked upload without Content-Length is cut off while streaming.
    chunked = client.post("/api/v1/builds", content=iter([b'{"request": "', b"x" * 2000, b'"}']),
                          headers={**BUILDER, "Content-Type": "application/json"})
    assert chunked.status_code == 413


def test_cors_allow_list(client):
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Authorization"}
    allowed = client.options("/api/v1/builds", headers={"Origin": "http://localhost:3000", **preflight})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    denied = client.options("/api/v1/builds", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in denied.headers


def test_internal_errors_are_not_leaked(client):
    def explode():
        raise RuntimeError(r"database at C:\secret\metaagent.db exploded with key sk-123")

    client.app.dependency_overrides[get_builds_service] = explode
    try:
        response = client.get("/api/v1/builds", headers=VIEWER)
    finally:
        client.app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_api_key_parsing_rejects_weak_or_malformed_entries():
    strong = "k" * 32
    keys = parse_api_keys(f"ops:builder:{strong}; bad-entry ; short:viewer:abc; who:admin:{strong}; ro:viewer:{strong}")
    assert [(k.name, k.role) for k in keys] == [("ops", "builder"), ("ro", "viewer")]
    assert strong not in repr(keys[0])  # keys never appear in reprs/logs


# ── architecture ──

FORBIDDEN_IN_ENGINE = ("fastapi", "starlette", "backend", "sqlmodel", "sqlalchemy", "alembic", "uvicorn",
                       "streamlit")


def test_engine_has_no_backend_or_web_framework_imports():
    offenders = []
    for path in (PROJECT_ROOT / "metaagent").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] in FORBIDDEN_IN_ENGINE]
    assert offenders == []


def test_engine_imports_without_loading_web_frameworks():
    code = ("import sys, metaagent.pipeline, metaagent.builds.service, metaagent.rag.service, "
            "metaagent.execution.runner; "
            f"bad = [m for m in sys.modules if m.split('.')[0] in {FORBIDDEN_IN_ENGINE!r}]; "
            "print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
