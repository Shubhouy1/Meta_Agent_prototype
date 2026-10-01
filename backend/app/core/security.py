"""API-key authentication with two roles.

    Authorization: Bearer <api key>

Keys are configured in METAAGENT_API_KEYS as "name:role:key;..." and compared
in constant time. Roles:

    viewer   read builds, events, deployments and the document list
    builder  everything a viewer can do, plus: create builds (runs generated
             code in test mode), invoke deployments (runs generated code with
             model access), upload/delete documents, query documents

Fails closed: with no keys configured, every protected endpoint returns 401.
There is no anonymous access to anything that executes generated code or
spends model quota.
"""

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import ApiSettings
from backend.app.core.errors import ApiError

ROLE_PERMISSIONS = {
    "viewer": frozenset({"read"}),
    "builder": frozenset({"read", "build", "invoke", "rag_write", "rag_query"}),
}

_bearer = HTTPBearer(auto_error=False, description="API key issued via METAAGENT_API_KEYS")


class Unauthenticated(ApiError):
    status_code = 401
    headers = {"WWW-Authenticate": "Bearer"}


class Forbidden(ApiError):
    status_code = 403


@dataclass(frozen=True)
class Principal:
    name: str
    role: str

    def can(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, frozenset())


def authenticate(settings: ApiSettings, token: Optional[str]) -> Principal:
    if not settings.api_keys:
        raise Unauthenticated("API authentication is not configured on this server")
    if not token:
        raise Unauthenticated("Missing API key")
    match = None
    for key in settings.api_keys:
        # Compare against every key so timing doesn't reveal which prefix matched.
        if hmac.compare_digest(key.key.encode(), token.encode()):
            match = key
    if match is None:
        raise Unauthenticated("Invalid API key")
    return Principal(name=match.name, role=match.role)


def current_principal(request: Request,
                      credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> Principal:
    settings: ApiSettings = request.app.state.container.settings
    token = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else None
    return authenticate(settings, token)


def require(permission: str):
    """Dependency factory: `principal = Depends(require("build"))`."""

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.can(permission):
            raise Forbidden(f"API key '{principal.name}' ({principal.role}) may not perform this action")
        return principal

    dependency.__name__ = f"require_{permission}"
    return dependency
