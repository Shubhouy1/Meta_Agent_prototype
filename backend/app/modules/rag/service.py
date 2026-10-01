"""RAG application service.

    route → RagService → metaagent DocumentService (ingest / list / delete / answer)

Workspaces are the isolation hook: each maps to one Chroma collection.
Phase 2 has a single workspace ("default" → the shared collection that
generated RAG agents read), so every API key sees the same documents.
Per-user isolation is NOT implemented yet.
"""

import logging
import threading
from typing import Callable, Dict, List, Sequence, Tuple

from backend.app.core.errors import InvalidRequest, NotFound, UpstreamError
from metaagent.ai.llm import describe_error
from metaagent.rag.service import DEFAULT_COLLECTION, Answer, DocumentInfo, DocumentService, IngestResult

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "default"
_WORKSPACE_COLLECTIONS = {DEFAULT_WORKSPACE: DEFAULT_COLLECTION}


class RagService:
    def __init__(self, document_service_factory: Callable[[str], DocumentService]):
        self._factory = document_service_factory
        self._services: Dict[str, DocumentService] = {}
        self._lock = threading.Lock()

    def _documents(self, workspace: str) -> DocumentService:
        collection = _WORKSPACE_COLLECTIONS.get(workspace)
        if collection is None:
            raise NotFound(f"Unknown workspace '{workspace}'")
        with self._lock:
            if workspace not in self._services:
                self._services[workspace] = self._factory(collection)
            return self._services[workspace]

    def upload(self, files: Sequence[Tuple[str, bytes]], workspace: str = DEFAULT_WORKSPACE) -> IngestResult:
        try:
            result = self._documents(workspace).ingest(files)
        except Exception as e:
            logger.warning("Document ingestion failed: %s", e)
            raise UpstreamError(f"Could not index the documents: {describe_error(e)}") from e
        if not result.files:
            raise InvalidRequest("No documents were indexed: " + "; ".join(result.rejected or ["no files"]))
        return result

    def list(self, workspace: str = DEFAULT_WORKSPACE) -> List[DocumentInfo]:
        return self._documents(workspace).list_documents()

    def delete(self, doc_id: str, workspace: str = DEFAULT_WORKSPACE) -> int:
        removed = self._documents(workspace).delete_document(doc_id)
        if not removed:
            raise NotFound("Document not found")
        return removed

    def query(self, question: str, workspace: str = DEFAULT_WORKSPACE) -> Answer:
        try:
            return self._documents(workspace).answer(question)
        except Exception as e:
            logger.warning("RAG query failed: %s", e)
            raise UpstreamError(f"The question could not be answered: {describe_error(e)}") from e
