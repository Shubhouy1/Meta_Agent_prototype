from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from metaagent.rag.service import Answer, DocumentInfo, IngestResult


class DocumentOut(BaseModel):
    id: str = Field(description="SHA-256 of the file contents; re-uploading the same file keeps the same id")
    filename: str
    chunks: int
    size_bytes: Optional[int]
    uploaded_at: Optional[str]


class UploadResponse(BaseModel):
    documents: List[DocumentOut]
    chunks: int
    rejected: List[str] = Field(description="Files that were skipped, with the reason")


class DocumentList(BaseModel):
    items: List[DocumentOut]
    workspace: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid",
                              json_schema_extra={"examples": [{"question": "What is this document about?"}]})

    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


def document_out(info: DocumentInfo) -> DocumentOut:
    return DocumentOut(id=info.id, filename=info.filename, chunks=info.chunks,
                       size_bytes=info.size_bytes, uploaded_at=info.uploaded_at)


def upload_response(result: IngestResult) -> UploadResponse:
    return UploadResponse(documents=[document_out(d) for d in result.documents], chunks=result.chunks,
                          rejected=result.rejected)


def query_response(answer: Answer) -> QueryResponse:
    return QueryResponse(answer=answer.text,
                         sources=[Source(source=s, excerpt=e) for s, e in answer.sources])
