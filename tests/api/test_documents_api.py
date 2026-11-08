from typing import Any

import pytest
from fastapi.testclient import TestClient

from doc_agent.container import Container
from tests.conftest import ALICE, BOB, auth


def upload(client: TestClient, name: str, data: bytes, token: str = ALICE) -> Any:
    return client.post("/api/documents", files={"file": (name, data)}, headers=auth(token))


def test_health_reports_backends(client: TestClient) -> None:
    assert client.get("/api/health").json() == {
        "status": "ok",
        "model": "heuristic",
        "embedder": "hashing:256",
        "database": "sqlite",
    }


def test_list_documents_is_per_user(client: TestClient) -> None:
    alice = client.get("/api/documents", headers=auth(ALICE)).json()
    bob = client.get("/api/documents", headers=auth(BOB)).json()
    assert {d["id"] for d in alice} == {"alice-msa", "alice-handbook", "alice-pricing"}
    assert {d["id"] for d in bob} == {"bob-msa", "bob-handbook", "bob-pricing"}
    assert {d["title"]: d["stale"] for d in alice}["pricing-2024.csv"] is True


def test_upload_then_ask_about_it(client: TestClient) -> None:
    res = upload(client, "notes.md", b"The office wifi password rotates every Monday at 9am.")
    assert res.status_code == 201
    assert res.json()["chunk_count"] == 1

    run = client.post(
        "/api/runs", json={"question": "When does the office wifi rotate?"}, headers=auth(ALICE)
    ).json()
    assert run["citations"][0]["document_title"] == "notes.md"


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("malware.exe", b"MZ...", "Unsupported file type"),
        ("empty.txt", b"", "empty"),
        ("blank.txt", b"   \n\t ", "No text"),
        ("bad.txt", b"\xff\xfe\xfa", "UTF-8"),
        ("broken.docx", b"this is not a zip file", "corrupt"),
        ("ragged.csv", b"a,b\n1,2,3\n", "row 2 has 3 columns"),
        ("big.txt", b"x" * (5 * 1024 * 1024 + 1), "larger than"),
    ],
)
def test_upload_rejects_bad_files(client: TestClient, name: str, data: bytes, message: str) -> None:
    res = upload(client, name, data)
    assert res.status_code == 400
    assert message in res.json()["detail"]


def test_upload_requires_auth(client: TestClient) -> None:
    assert client.post("/api/documents", files={"file": ("a.txt", b"hi")}).status_code == 401


def test_embedding_outage_returns_503(
    client: TestClient, container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openai

    def down(*_: Any) -> Any:
        raise openai.APIConnectionError(request=None)  # type: ignore[arg-type]

    monkeypatch.setattr(container.embedder, "embed", down)
    assert upload(client, "a.txt", b"Some text.").status_code == 503
