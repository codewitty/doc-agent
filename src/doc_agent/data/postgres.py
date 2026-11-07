"""Postgres + pgvector adapters for the deployed site (Supabase).

Tables live in their own `doc_agent` schema so they can share a Supabase
project with other apps. The embedding column has no fixed dimension, so
the same schema works for any embedder; that rules out an ANN index, which
is an acceptable trade-off at demo scale.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from doc_agent.domain import AgentRun, ChunkHit, Document, NewChunk

SCHEMA = """
create extension if not exists vector;
create schema if not exists doc_agent;
create table if not exists doc_agent.documents (
  id text primary key,
  user_id text not null,
  title text not null,
  mime_type text not null,
  created_at timestamptz not null
);
create table if not exists doc_agent.chunks (
  id text primary key,
  document_id text not null references doc_agent.documents(id) on delete cascade,
  user_id text not null,
  chunk_index integer not null,
  content text not null,
  embedding vector not null
);
create index if not exists chunks_user_idx on doc_agent.chunks (user_id);
create table if not exists doc_agent.agent_runs (
  id text primary key,
  user_id text not null,
  status text not null,
  state jsonb not null,
  updated_at timestamptz not null
);
"""


class PostgresDatabase:
    def __init__(self, url: str) -> None:
        # The schema has to exist before register_vector can find the type
        with Connection.connect(url, autocommit=True) as conn:
            conn.execute(SCHEMA)
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
            configure=register_vector,
            open=True,
        )

    def close(self) -> None:
        self.pool.close()


class PostgresDocumentRepository:
    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db

    def add_document(
        self,
        user_id: str,
        title: str,
        mime_type: str,
        chunks: list[NewChunk],
        created_at: datetime,
        document_id: str | None = None,
    ) -> Document:
        doc_id = document_id or str(uuid.uuid4())
        with self.db.pool.connection() as conn, conn.transaction():
            conn.execute(
                "insert into doc_agent.documents (id, user_id, title, mime_type, created_at) "
                "values (%s, %s, %s, %s, %s)",
                (doc_id, user_id, title, mime_type, created_at),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into doc_agent.chunks "
                    "(id, document_id, user_id, chunk_index, content, embedding) "
                    "values (%s, %s, %s, %s, %s, %s::vector)",
                    [
                        (
                            f"{doc_id}-c{c.index}",
                            doc_id,
                            user_id,
                            c.index,
                            c.content,
                            str(c.embedding),
                        )
                        for c in chunks
                    ],
                )
        return Document(
            id=doc_id,
            user_id=user_id,
            title=title,
            mime_type=mime_type,
            created_at=created_at,
            chunk_count=len(chunks),
        )

    def list_documents(self, user_id: str) -> list[Document]:
        with self.db.pool.connection() as conn:
            rows: list[dict[str, Any]] = conn.execute(
                "select d.*, count(c.id)::int as chunk_count from doc_agent.documents d "
                "left join doc_agent.chunks c on c.document_id = d.id "
                "where d.user_id = %s group by d.id order by d.created_at desc",
                (user_id,),
            ).fetchall()
        return [Document(**r) for r in rows]

    def search(self, user_id: str, embedding: list[float], top_k: int) -> list[ChunkHit]:
        with self.db.pool.connection() as conn:
            rows: list[dict[str, Any]] = conn.execute(
                "select c.id as chunk_id, c.document_id, c.content, d.title as document_title, "
                "d.created_at as document_created_at, "
                "greatest(-1, least(1, 1 - (c.embedding <=> %s::vector))) as similarity "
                "from doc_agent.chunks c join doc_agent.documents d on d.id = c.document_id "
                "where c.user_id = %s and vector_dims(c.embedding) = %s "
                "order by c.embedding <=> %s::vector limit %s",
                (str(embedding), user_id, len(embedding), str(embedding), top_k),
            ).fetchall()
        return [ChunkHit(**r) for r in rows]

    def delete_document(self, user_id: str, document_id: str) -> bool:
        with self.db.pool.connection() as conn:
            cur = conn.execute(
                "delete from doc_agent.documents where id = %s and user_id = %s",
                (document_id, user_id),
            )
        return cur.rowcount > 0


class PostgresRunRepository:
    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db

    def create(self, run: AgentRun) -> None:
        with self.db.pool.connection() as conn:
            conn.execute(
                "insert into doc_agent.agent_runs (id, user_id, status, state, updated_at) "
                "values (%s, %s, %s, %s, %s)",
                (
                    run.id,
                    run.user_id,
                    run.status,
                    Jsonb(run.model_dump(mode="json")),
                    run.updated_at,
                ),
            )

    def get(self, user_id: str, run_id: str) -> AgentRun | None:
        with self.db.pool.connection() as conn:
            row: dict[str, Any] | None = conn.execute(
                "select state from doc_agent.agent_runs where id = %s and user_id = %s",
                (run_id, user_id),
            ).fetchone()
        return AgentRun.model_validate(row["state"]) if row else None

    def save(self, run: AgentRun) -> None:
        with self.db.pool.connection() as conn:
            conn.execute(
                "update doc_agent.agent_runs set status = %s, state = %s, updated_at = %s "
                "where id = %s and user_id = %s",
                (
                    run.status,
                    Jsonb(run.model_dump(mode="json")),
                    run.updated_at,
                    run.id,
                    run.user_id,
                ),
            )
