import pytest

from metaagent.ai.generator.generator import CodeGenerator
from metaagent.schemas import CodeGenerationResult
from tests.backend.conftest import BUILDER, CHATBOT_PLAN, VIEWER, start_build, wait_for_build
from tests.conftest import CRASHING_AGENT, FakeLLM, fenced

pytestmark = pytest.mark.subprocess

# Passes Stage 4 (sample input), but raises for one specific input at invoke time.
MOODY_AGENT = '''
class Agent:
    def run(self, user_input: str) -> str:
        if user_input == "explode":
            raise ValueError("exploded on request")
        return f"Echo: {user_input}"
'''


@pytest.fixture
def deployed(client, fake_llm):
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(MOODY_AGENT)])
    build_id = start_build(client)
    build = wait_for_build(client, build_id)
    assert build["deployed"]
    return build["deployment"]["deployment_id"]


def test_get_deployment(client, deployed):
    response = client.get(f"/api/v1/deployments/{deployed}", headers=VIEWER)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready" and body["build_id"] == deployed
    assert {c["name"] for c in body["checks"]} == {"syntax", "policy", "structure", "runtime", "output"}
    assert "deployed_file" not in body and "run_command" not in body  # no server paths


def test_invoke_runs_the_agent(client, deployed):
    response = client.post(f"/api/v1/deployments/{deployed}/invoke", json={"input": "hello"}, headers=BUILDER)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] and body["output"] == "Echo: hello" and not body["timed_out"]


def test_agent_failure_is_reported_in_the_body(client, deployed):
    body = client.post(f"/api/v1/deployments/{deployed}/invoke", json={"input": "explode"},
                       headers=BUILDER).json()
    assert body["ok"] is False and body["output"] is None
    assert "ValueError" in body["error"] and "exploded on request" in body["error"]


def test_viewer_cannot_invoke(client, deployed):
    response = client.post(f"/api/v1/deployments/{deployed}/invoke", json={"input": "hi"}, headers=VIEWER)
    assert response.status_code == 403


def test_unknown_deployment(client):
    assert client.get("/api/v1/deployments/nope", headers=VIEWER).status_code == 404
    response = client.post("/api/v1/deployments/nope/invoke", json={"input": "hi"}, headers=BUILDER)
    assert response.status_code == 404 and response.json() == {"detail": "Deployment not found"}


def test_failed_build_has_no_deployment_and_cannot_be_invoked(client, fake_llm, monkeypatch):
    monkeypatch.setattr(CodeGenerator, "generate_template",
                        lambda self, plan, sel, req: CodeGenerationResult(code=CRASHING_AGENT, method="template", lines=3))
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(CRASHING_AGENT), fenced(CRASHING_AGENT)])
    build_id = start_build(client)
    assert wait_for_build(client, build_id)["status"] == "failed"

    assert client.get(f"/api/v1/deployments/{build_id}", headers=VIEWER).status_code == 404
    response = client.post(f"/api/v1/deployments/{build_id}/invoke", json={"input": "hi"}, headers=BUILDER)
    assert response.status_code == 409
    assert "not deployed" in response.json()["detail"] and "testing" in response.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"input": ""}, {"input": "x" * 4001}, {"input": "hi", "extra": 1}])
def test_invalid_invoke_requests(client, deployed, body):
    assert client.post(f"/api/v1/deployments/{deployed}/invoke", json=body, headers=BUILDER).status_code == 422


def test_invocation_rate_limit(make_client, fake_llm):
    client = make_client(invocations_per_minute=1)
    fake_llm.current = FakeLLM(plan=CHATBOT_PLAN, code=[fenced(MOODY_AGENT)])
    dep = wait_for_build(client, start_build(client))["deployment"]["deployment_id"]
    assert client.post(f"/api/v1/deployments/{dep}/invoke", json={"input": "a"}, headers=BUILDER).status_code == 200
    limited = client.post(f"/api/v1/deployments/{dep}/invoke", json={"input": "b"}, headers=BUILDER)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
