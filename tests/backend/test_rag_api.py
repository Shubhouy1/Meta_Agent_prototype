from metaagent.core.config import Settings
from tests.backend.conftest import BUILDER, VIEWER

DOC = ("MetaAgent turns a plain-English request into a tested agent. " * 30).encode()


def upload(client, *files, headers=BUILDER):
    return client.post("/api/v1/rag/documents", headers=headers,
                       files=[("files", (name, data, "text/plain")) for name, data in files])


def test_upload_list_query_delete(client):
    response = upload(client, ("guide.txt", DOC))
    assert response.status_code == 201, response.text
    body = response.json()
    doc_id = body["documents"][0]["id"]
    assert body["chunks"] > 1 and body["rejected"] == []

    listed = client.get("/api/v1/rag/documents", headers=VIEWER).json()
    assert listed["workspace"] == "default"
    assert [(d["id"], d["filename"]) for d in listed["items"]] == [(doc_id, "guide.txt")]

    answer = client.post("/api/v1/rag/query", json={"question": "What does MetaAgent do?"}, headers=BUILDER)
    assert answer.status_code == 200
    assert answer.json()["answer"] == "answer from context"
    assert answer.json()["sources"][0]["source"] == "guide.txt"

    assert client.delete(f"/api/v1/rag/documents/{doc_id}", headers=BUILDER).status_code == 204
    assert client.get("/api/v1/rag/documents", headers=VIEWER).json()["items"] == []
    assert client.delete(f"/api/v1/rag/documents/{doc_id}", headers=BUILDER).status_code == 404


def test_reupload_does_not_duplicate(client):
    first = upload(client, ("guide.txt", DOC)).json()
    upload(client, ("guide.txt", DOC))
    items = client.get("/api/v1/rag/documents", headers=VIEWER).json()["items"]
    assert len(items) == 1 and items[0]["chunks"] == first["chunks"]


def test_partial_upload_reports_rejected_files(client):
    body = upload(client, ("guide.txt", DOC), ("tool.exe", b"MZ")).json()
    assert len(body["documents"]) == 1 and "tool.exe" in body["rejected"][0]


def test_unsupported_or_empty_uploads_are_rejected(client):
    response = upload(client, ("tool.exe", b"MZ"), ("empty.txt", b""))
    assert response.status_code == 422 and "No documents were indexed" in response.json()["detail"]


def test_upload_limits(make_client, settings):
    small = make_client(engine=Settings(data_dir=settings.data_dir, max_upload_mb=1), max_upload_files=2)
    too_big = upload(small, ("big.txt", b"x" * (1024 * 1024 + 10)))
    assert too_big.status_code == 413
    too_many = upload(small, ("a.txt", DOC), ("b.txt", DOC), ("c.txt", DOC))
    assert too_many.status_code == 422 and "Too many files" in too_many.json()["detail"]


def test_query_validation(client):
    assert client.post("/api/v1/rag/query", json={"question": ""}, headers=BUILDER).status_code == 422
    assert client.post("/api/v1/rag/query", json={"question": "x" * 2001}, headers=BUILDER).status_code == 422


def test_viewer_can_list_but_not_change_or_query(client):
    assert client.get("/api/v1/rag/documents", headers=VIEWER).status_code == 200
    assert upload(client, ("guide.txt", DOC), headers=VIEWER).status_code == 403
    assert client.delete("/api/v1/rag/documents/abc", headers=VIEWER).status_code == 403
    assert client.post("/api/v1/rag/query", json={"question": "hi"}, headers=VIEWER).status_code == 403
