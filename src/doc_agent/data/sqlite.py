"""SQLite adapters for local development, tests and CI.

Vector search loads the user's chunk embeddings and ranks them in Python.
That is fine for hundreds of chunks; the Postgres adapter uses pgvector.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from doc_agent.data.vectors import cosine
from doc_agent.domain import AgentRun, ChunkHit, Document, NewChunk

SCHEMA = """
create table if not exists documents (
  id text primary key,
  user_id text not null,
  title text not null,
  mime_type text not null,
  created_at text not null
);
create table if not exists chunks (
  id text primary key,
  document_id text not null references documents(id) on delete cascade,
  user_id text not null,
  chunk_index integer not null,
  content text not null,
  embedding text not null
);
create index if not exists chunks_user_idx on chunks (user_id);
create table if not exists agent_runs (
  id text primary key,
  user_id text not null,
  status text not null,
  state text not null,
  updated_at text not null
);
"""


class SQLiteDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # A connection per operation keeps this safe to call from worker threads
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        try:
            with conn:
                yield conn
        finally:
            conn.close()


class SQLiteDocumentRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
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
        with self.db.connect() as db:
            db.execute(
                "insert into documents (id, user_id, title, mime_type, created_at) values (?, ?, ?, ?, ?)",
                (doc_id, user_id, title, mime_type, created_at.isoformat()),
            )
            db.executemany(
                "insert into chunks (id, document_id, user_id, chunk_index, content, embedding) "
                "values (?, ?, ?, ?, ?, ?)",
                [
                    (
                        f"{doc_id}-c{c.index}",
                        doc_id,
                        user_id,
                        c.index,
                        c.content,
                        json.dumps(c.embedding),
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
        with self.db.connect() as db:
            rows = db.execute(
                "select d.*, count(c.id) as chunk_count from documents d "
                "left join chunks c on c.document_id = d.id "
                "where d.user_id = ? group by d.id order by d.created_at desc",
                (user_id,),
            ).fetchall()
        return [
            Document(
                id=r["id"],
                user_id=r["user_id"],
                title=r["title"],
                mime_type=r["mime_type"],
                created_at=datetime.fromisoformat(r["created_at"]),
                chunk_count=r["chunk_count"],
            )
            for r in rows
        ]

    def search(self, user_id: str, embedding: list[float], top_k: int) -> list[ChunkHit]:
        with self.db.connect() as db:
            rows = db.execute(
                "select c.id, c.document_id, c.content, c.embedding, d.title, d.created_at "
                "from chunks c join documents d on d.id = c.document_id where c.user_id = ?",
                (user_id,),
            ).fetchall()
        # Chunks embedded by a different embedder have other dimensions and
        # cannot be compared, so they are skipped (same rule as Postgres)
        hits = [
            ChunkHit(
                chunk_id=r["id"],
                document_id=r["document_id"],
                document_title=r["title"],
                document_created_at=datetime.fromisoformat(r["created_at"]),
                content=r["content"],
                similarity=cosine(embedding, json.loads(r["embedding"])),
            )
            for r in rows
            if len(json.loads(r["embedding"])) == len(embedding)
        ]
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:top_k]

    def delete_document(self, user_id: str, document_id: str) -> bool:
        with self.db.connect() as db:
            cur = db.execute(
                "delete from documents where id = ? and user_id = ?", (document_id, user_id)
            )
        return cur.rowcount > 0


class SQLiteRunRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def create(self, run: AgentRun) -> None:
        with self.db.connect() as db:
            db.execute(
                "insert into agent_runs (id, user_id, status, state, updated_at) values (?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.user_id,
                    run.status,
                    run.model_dump_json(),
                    run.updated_at.isoformat(),
                ),
            )

    def get(self, user_id: str, run_id: str) -> AgentRun | None:
        with self.db.connect() as db:
            row = db.execute(
                "select state from agent_runs where id = ? and user_id = ?", (run_id, user_id)
            ).fetchone()
        return AgentRun.model_validate_json(row["state"]) if row else None

    def save(self, run: AgentRun) -> None:
        with self.db.connect() as db:
            db.execute(
                "update agent_runs set status = ?, state = ?, updated_at = ? where id = ? and user_id = ?",
                (
                    run.status,
                    run.model_dump_json(),
                    run.updated_at.isoformat(),
                    run.id,
                    run.user_id,
                ),
            )
