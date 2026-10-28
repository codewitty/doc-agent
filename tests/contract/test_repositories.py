"""Both storage adapters must behave the same. SQLite always runs; Postgres
runs when DATABASE_URL_TEST points at a database with pgvector (CI uses a
service container). Each Postgres test drops the doc_agent schema first."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from doc_agent.data.sqlite import SQLiteDatabase, SQLiteDocumentRepository, SQLiteRunRepository
from doc_agent.domain import AgentRun, Message, NewChunk, RunStatus
from doc_agent.ports import DocumentRepository, RunRepository

T0 = datetime(2026, 1, 1, tzinfo=UTC)
PG_URL = os.environ.get("DATABASE_URL_TEST")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def repos(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[DocumentRepository, RunRepository]]:
    if request.param == "sqlite":
        db = SQLiteDatabase(tmp_path / "contract.db")
        yield SQLiteDocumentRepository(db), SQLiteRunRepository(db)
        return

    if not PG_URL:
        pytest.skip("DATABASE_URL_TEST not set")
    import psycopg

    from doc_agent.data.postgres import (
        PostgresDatabase,
        PostgresDocumentRepository,
        PostgresRunRepository,
    )

    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute("drop schema if exists doc_agent cascade")
    pg = PostgresDatabase(PG_URL)
    yield PostgresDocumentRepository(pg), PostgresRunRepository(pg)
    pg.close()


def chunks(*vectors: list[float]) -> list[NewChunk]:
    return [NewChunk(index=i, content=f"chunk {i}", embedding=v) for i, v in enumerate(vectors)]


def test_add_and_list_documents(repos: Any) -> None:
    docs, _ = repos
    docs.add_document("u1", "a.txt", "text/plain", chunks([1, 0], [0, 1]), T0, document_id="d1")
    docs.add_document(
        "u1", "b.txt", "text/plain", chunks([1, 1]), T0 + timedelta(days=1), document_id="d2"
    )
    docs.add_document("u2", "c.txt", "text/plain", chunks([1, 0]), T0, document_id="d3")

    listed = docs.list_documents("u1")
    assert [d.id for d in listed] == ["d2", "d1"]  # newest first
    assert [d.chunk_count for d in listed] == [1, 2]
    assert listed[1].created_at == T0


def test_search_ranks_by_similarity_and_is_tenant_scoped(repos: Any) -> None:
    docs, _ = repos
    docs.add_document(
        "u1",
        "a.txt",
        "text/plain",
        chunks([1, 0, 0], [0, 1, 0], [0.7, 0.7, 0]),
        T0,
        document_id="d1",
    )
    docs.add_document("u2", "b.txt", "text/plain", chunks([1, 0, 0]), T0, document_id="d2")

    hits = docs.search("u1", [1, 0, 0], top_k=2)
    assert [h.chunk_id for h in hits] == ["d1-c0", "d1-c2"]
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[0].document_title == "a.txt"
    assert all(h.document_id == "d1" for h in docs.search("u1", [1, 0, 0], top_k=10))


def test_search_skips_embeddings_with_other_dimensions(repos: Any) -> None:
    docs, _ = repos
    docs.add_document("u1", "a.txt", "text/plain", chunks([1, 0]), T0, document_id="d1")
    docs.add_document("u1", "b.txt", "text/plain", chunks([1, 0, 0]), T0, document_id="d2")
    assert [h.document_id for h in docs.search("u1", [1, 0, 0], top_k=10)] == ["d2"]


def test_delete_is_tenant_scoped_and_cascades(repos: Any) -> None:
    docs, _ = repos
    docs.add_document("u1", "a.txt", "text/plain", chunks([1, 0]), T0, document_id="d1")
    assert docs.delete_document("u2", "d1") is False
    assert docs.delete_document("u1", "d1") is True
    assert docs.delete_document("u1", "d1") is False
    assert docs.search("u1", [1, 0], top_k=5) == []


def test_run_round_trip_and_tenant_scope(repos: Any) -> None:
    _, runs = repos
    run = AgentRun(
        id="r1",
        user_id="u1",
        question="q",
        status=RunStatus.RUNNING,
        messages=[Message(role="user", content="q")],
        created_at=T0,
        updated_at=T0,
    )
    runs.create(run)
    run.status = RunStatus.COMPLETED
    run.answer = "a"
    runs.save(run)

    loaded = runs.get("u1", "r1")
    assert loaded == run
    assert runs.get("u2", "r1") is None
    assert runs.get("u1", "missing") is None
