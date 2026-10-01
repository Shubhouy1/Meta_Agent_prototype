"""Deployment endpoints."""

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_deployments_service, rate_limit
from backend.app.core.security import Principal, require
from backend.app.modules.builds.api.schemas import API_PREFIX
from backend.app.modules.deployments.api.schemas import (
    DeploymentOut, InvokeRequest, InvokeResponse, deployment_out, invoke_response,
)
from backend.app.modules.deployments.service import DeploymentsService

router = APIRouter(prefix=f"{API_PREFIX}/deployments", tags=["deployments"])

_ERRORS = {401: {"description": "Missing or invalid API key"},
           403: {"description": "API key lacks permission"},
           404: {"description": "Deployment not found"}}


@router.get("/{deployment_id}", response_model=DeploymentOut, summary="Get a deployment",
            description="A deployed agent: the build it came from and the checks it passed.",
            responses=_ERRORS)
def get_deployment(deployment_id: str, _: Principal = Depends(require("read")),
                   service: DeploymentsService = Depends(get_deployments_service)) -> DeploymentOut:
    record, result = service.get(deployment_id)
    return deployment_out(record, result)


@router.post(
    "/{deployment_id}/invoke", response_model=InvokeResponse, summary="Invoke a deployed agent",
    description=(
        "Runs the agent once with the given input and returns its output.\n\n"
        "The agent runs in a separate process with a timeout and without API keys; model calls are made "
        "by the server on its behalf. **This is not a hardened sandbox**: generated code is not isolated "
        "from the server's filesystem or network, so invocation requires the **builder** role.\n\n"
        "A failing agent returns 200 with `ok: false` and an `error`; HTTP errors are reserved for "
        "problems with the request itself."
    ),
    responses={**_ERRORS, 409: {"description": "The build exists but was not deployed (it failed)"},
               429: {"description": "Rate limit or concurrent invocation limit exceeded"}},
)
def invoke_deployment(deployment_id: str, body: InvokeRequest,
                      _: Principal = Depends(require("invoke")),
                      __: None = Depends(rate_limit("invocations", "invocations_per_minute", 60)),
                      service: DeploymentsService = Depends(get_deployments_service)) -> InvokeResponse:
    return invoke_response(deployment_id, service.invoke(deployment_id, body.input))
