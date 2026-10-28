from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from doc_agent.agent.tools import (
    DELETE_DOCUMENT,
    LIST_DOCUMENTS,
    SEARCH_DOCUMENTS,
    SearchArgs,
    ToolContext,
    is_stale,
)
from doc_agent.container import Container
from tests.conftest import NOW


def ctx(container: Container, user: str = "alice") -> ToolContext:
    return ToolContext(
        user_id=user, documents=container.documents, embedder=container.embedder, now=NOW
    )


@pytest.mark.parametrize(
    "args",
    [{"query": ""}, {"query": "x", "top_k": 50}, {"query": "x", "extra": 1}, "not an object"],
)
def test_search_rejects_bad_arguments(args: Any) -> None:
    with pytest.raises(ValidationError):
        SearchArgs.model_validate(args)


def test_search_defaults_top_k() -> None:
    assert SearchArgs.model_validate({"query": "payment"}).top_k == 5


def test_search_only_returns_callers_chunks(container: Container) -> None:
    out = SEARCH_DOCUMENTS.run(SearchArgs(query="payment due invoice", top_k=10), ctx(container))
    assert out.hits
    assert all(h.document_id.startswith("alice-") for h in out.hits)


def test_search_marks_old_documents_stale(container: Container) -> None:
    out = SEARCH_DOCUMENTS.run(SearchArgs(query="Growth plan monthly price"), ctx(container))
    top = out.content["results"][0]
    assert top["document"] == "pricing-2024.csv"
    assert top["stale"] is True


def test_search_rejects_malformed_repository_response(container: Container) -> None:
    class BrokenRepo:
        def search(self, *_: Any) -> list[dict[str, Any]]:
            return [{"chunk_id": "", "similarity": 7}]

    broken = ToolContext(
        user_id="alice",
        documents=BrokenRepo(),  # type: ignore[arg-type]
        embedder=container.embedder,
        now=NOW,
    )
    with pytest.raises(ValidationError):
        SEARCH_DOCUMENTS.run(SearchArgs(query="x"), broken)


def test_list_documents_reports_staleness(container: Container) -> None:
    docs = LIST_DOCUMENTS.run(LIST_DOCUMENTS.args(), ctx(container)).content["documents"]
    assert {d["title"]: d["stale"] for d in docs} == {
        "service-agreement.docx": False,
        "employee-handbook.docx": False,
        "pricing-2024.csv": True,
    }


def test_delete_requires_approval() -> None:
    assert DELETE_DOCUMENT.requires_approval
    assert not SEARCH_DOCUMENTS.requires_approval


def test_delete_cannot_touch_another_users_document(container: Container) -> None:
    args = DELETE_DOCUMENT.args.model_validate({"document_id": "bob-pricing"})
    assert DELETE_DOCUMENT.run(args, ctx(container)).content == {"error": "not_found"}
    assert any(d.id == "bob-pricing" for d in container.documents.list_documents("bob"))


def test_stale_threshold_is_180_days() -> None:
    assert not is_stale(NOW - timedelta(days=180), NOW)
    assert is_stale(NOW - timedelta(days=181), NOW)


def test_tool_specs_are_valid_json_schema_objects() -> None:
    for tool in (SEARCH_DOCUMENTS, LIST_DOCUMENTS, DELETE_DOCUMENT):
        assert tool.spec.parameters["type"] == "object"
