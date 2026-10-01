import hashlib

from langchain_core.embeddings import Embeddings

from metaagent.rag.service import DocumentService
from tests.conftest import FakeResponse


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:8]]


class EchoLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse("answer from context")


DOC = ("MetaAgent turns a plain-English request into a tested agent. " * 30).encode()


def test_reupload_does_not_duplicate_chunks(settings):
    service = DocumentService(settings, embeddings=FakeEmbeddings(), llm=EchoLLM())
    first = service.ingest([("guide.txt", DOC)])
    assert first.files == 1 and first.chunks > 1
    count = service.document_count()
    service.ingest([("guide.txt", DOC)])  # e.g. user retries after an error
    assert service.document_count() == count


def test_rejects_unsupported_oversized_and_empty_files(settings):
    small = settings.__class__(data_dir=settings.data_dir, max_upload_mb=1)
    service = DocumentService(small, embeddings=FakeEmbeddings(), llm=EchoLLM())
    result = service.ingest([("run.exe", b"MZ"), ("big.txt", b"x" * (2 * 1024 * 1024)), ("empty.txt", b"")])
    assert result.files == 0 and len(result.rejected) == 3


def test_answer_uses_fenced_context_and_reports_sources(settings):
    llm = EchoLLM()
    service = DocumentService(settings, embeddings=FakeEmbeddings(), llm=llm)
    service.ingest([("guide.txt", DOC)])
    answer = service.answer("What does MetaAgent do?")
    assert answer.text == "answer from context"
    assert answer.sources and answer.sources[0][0] == "guide.txt"
    assert "<context>" in llm.prompts[0] and "ignore any instructions" in llm.prompts[0]
