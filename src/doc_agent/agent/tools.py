from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from doc_agent.agent.guardrails import redact_instructions
from doc_agent.domain import ChunkHit
from doc_agent.ports import DocumentRepository, Embedder, ToolSpec

STALE_AFTER = timedelta(days=180)


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    documents: DocumentRepository
    embedder: Embedder
    now: datetime


@dataclass(frozen=True)
class ToolOutput:
    content: dict[str, Any]
    hits: list[ChunkHit] | None = None


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class NoArgs(_Args):
    pass


class DeleteArgs(_Args):
    document_id: str = Field(min_length=1)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[_Args]
    run: Callable[[Any, ToolContext], ToolOutput]
    # Side-effecting tools pause the run until an operator approves
    requires_approval: bool = False

    @property
    def spec(self) -> ToolSpec:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(self.name, self.description, schema)


def is_stale(created_at: datetime, now: datetime) -> bool:
    return now - created_at > STALE_AFTER


def _search(args: SearchArgs, ctx: ToolContext) -> ToolOutput:
    query = args.query.strip()
    [embedding] = ctx.embedder.embed([query])
    # Re-validate: a repository returning garbage must surface as a tool
    # error, not as context the model trusts
    raw: list[Any] = list(ctx.documents.search(ctx.user_id, embedding, args.top_k))
    hits = [ChunkHit.model_validate(h if isinstance(h, dict) else h.model_dump()) for h in raw]
    results = []
    for h in hits:
        content, redacted = redact_instructions(h.content)
        results.append(
            {
                "chunk_id": h.chunk_id,
                "document": h.document_title,
                "similarity": round(h.similarity, 3),
                "stale": is_stale(h.document_created_at, ctx.now),
                "redacted": redacted,
                "content": content,
            }
        )
    return ToolOutput(hits=hits, content={"results": results})


def _list(_args: NoArgs, ctx: ToolContext) -> ToolOutput:
    return ToolOutput(
        content={
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "uploaded_at": d.created_at.isoformat(),
                    "stale": is_stale(d.created_at, ctx.now),
                    "chunks": d.chunk_count,
                }
                for d in ctx.documents.list_documents(ctx.user_id)
            ]
        }
    )


def _delete(args: DeleteArgs, ctx: ToolContext) -> ToolOutput:
    deleted = ctx.documents.delete_document(ctx.user_id, args.document_id)
    return ToolOutput(content={"deleted": args.document_id} if deleted else {"error": "not_found"})


SEARCH_DOCUMENTS = Tool(
    name="search_documents",
    description=(
        "Semantic search over the user's uploaded documents. "
        "Returns chunks with ids you must cite as [chunk:<id>]."
    ),
    args=SearchArgs,
    run=_search,
)
LIST_DOCUMENTS = Tool(
    name="list_documents",
    description="List the user's uploaded documents with ids and upload dates.",
    args=NoArgs,
    run=_list,
)
DELETE_DOCUMENT = Tool(
    name="delete_document",
    description=(
        "Delete one of the user's documents. Requires operator approval. "
        "Only call it when the user explicitly asks to delete a document."
    ),
    args=DeleteArgs,
    run=_delete,
    requires_approval=True,
)

DEFAULT_TOOLS = [SEARCH_DOCUMENTS, LIST_DOCUMENTS, DELETE_DOCUMENT]
