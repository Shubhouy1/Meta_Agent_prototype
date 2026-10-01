"""Consistent API errors: every error response is {"detail": "<message>"}
(validation errors add an "errors" list). Internal details such as stack
traces, filesystem paths and raw provider payloads never reach clients."""

import logging
import re
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiError(Exception):
    status_code = 400
    headers: Optional[dict] = None

    def __init__(self, detail: str, headers: Optional[dict] = None):
        super().__init__(detail)
        self.detail = detail
        if headers:
            self.headers = headers


class NotFound(ApiError):
    status_code = 404


class Conflict(ApiError):
    status_code = 409


class InvalidRequest(ApiError):
    status_code = 422


class TooManyRequests(ApiError):
    status_code = 429


class ServiceUnavailable(ApiError):
    status_code = 503


class UpstreamError(ApiError):
    """The model provider failed (quota, auth, timeout)."""
    status_code = 502


# Absolute Windows paths (C:\..., \\server\...) and common POSIX roots.
_PATH_PATTERN = re.compile(
    r"([A-Za-z]:[\\/][^\s'\"<>|,;)]*|\\\\[^\s'\"<>|,;)]+"
    r"|/(?:home|Users|tmp|var|usr|opt|srv|mnt|root|app|private)/[^\s'\"<>|,;)]*)"
)


def redact(text: Optional[str]) -> Optional[str]:
    """Remove absolute filesystem paths from text that is returned to clients."""
    if text is None:
        return None
    return _PATH_PATTERN.sub("<path>", text)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse({"detail": "Invalid request", "errors": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
