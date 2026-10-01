"""RAG endpoints (single shared workspace in Phase 2)."""

from typing import List

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Request, Response, UploadFile

from backend.app.core.dependencies import get_container, get_rag_service, rate_limit
from backend.app.core.errors import ApiError, InvalidRequest
from backend.app.core.security import Principal, require
from backend.app.modules.builds.api.schemas import API_PREFIX
from backend.app.modules.rag.api.schemas import (
    DocumentList, QueryRequest, QueryResponse, UploadResponse, document_out, query_response, upload_response,
)
from backend.app.modules.rag.service import DEFAULT_WORKSPACE, RagService

router = APIRouter(prefix=f"{API_PREFIX}/rag", tags=["rag"])

_ERRORS = {401: {"description": "Missing or invalid API key"},
           403: {"description": "API key lacks permission"}}
_SHARED_NOTE = ("All API keys currently share one knowledge base (the one generated RAG agents read); "
                "per-user isolation is not implemented yet.")


class PayloadTooLarge(ApiError):
    status_code = 413


@router.post(
    "/documents", status_code=201, response_model=UploadResponse, summary="Upload documents",
    description=f"Index PDF or TXT files (multipart field `files`). Re-uploading a file replaces its chunks "
                f"instead of duplicating them. {_SHARED_NOTE}",
    responses={**_ERRORS, 413: {"description": "A file or the request is too large"},
               422: {"description": "No file could be indexed"},
               502: {"description": "Embedding provider error"}},
)
async def upload_documents(request: Request, files: List[UploadFile] = File(..., description="PDF or TXT files"),
                           _: Principal = Depends(require("rag_write")),
                           __: None = Depends(rate_limit("rag_uploads", "rag_uploads_per_hour", 3600)),
                           service: RagService = Depends(get_rag_service)) -> UploadResponse:
    settings = get_container(request).settings
    if len(files) > settings.max_upload_files:
        raise InvalidRequest(f"Too many files ({len(files)}); the limit is {settings.max_upload_files}")
    per_file = settings.engine.max_upload_bytes
    payload = []
    for upload in files:
        data = await upload.read(per_file + 1)
        if len(data) > per_file:
            raise PayloadTooLarge(f"{upload.filename}: larger than {settings.engine.max_upload_mb} MB")
        payload.append((upload.filename or "unnamed", data))
    return upload_response(await to_thread.run_sync(service.upload, payload))


@router.get("/documents", response_model=DocumentList, summary="List documents",
            description=_SHARED_NOTE, responses=_ERRORS)
def list_documents(_: Principal = Depends(require("read")),
                   service: RagService = Depends(get_rag_service)) -> DocumentList:
    return DocumentList(items=[document_out(d) for d in service.list()], workspace=DEFAULT_WORKSPACE)


@router.delete("/documents/{doc_id}", status_code=204, summary="Delete a document",
               description="Removes every chunk of the document from the knowledge base.",
               responses={**_ERRORS, 404: {"description": "Document not found"}})
def delete_document(doc_id: str, _: Principal = Depends(require("rag_write")),
                    service: RagService = Depends(get_rag_service)) -> Response:
    service.delete(doc_id)
    return Response(status_code=204)


@router.post("/query", response_model=QueryResponse, summary="Ask a question",
             description=f"Answers from the indexed documents only, with the excerpts used. {_SHARED_NOTE}",
             responses={**_ERRORS, 429: {"description": "Rate limit exceeded"},
                        502: {"description": "Model provider error (e.g. quota exceeded)"}})
def query_documents(body: QueryRequest, _: Principal = Depends(require("rag_query")),
                    __: None = Depends(rate_limit("rag_queries", "rag_queries_per_minute", 60)),
                    service: RagService = Depends(get_rag_service)) -> QueryResponse:
    return query_response(service.query(body.question))
