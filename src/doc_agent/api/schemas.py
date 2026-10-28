"""Request and response shapes. Raw model messages never leave the server."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from doc_agent.domain import AgentRun, Citation, Document, RunError, RunStatus, TraceStep


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Strict: lax mode would accept "yes" or 1 as an approval
    approve: StrictBool
    note: str | None = Field(default=None, max_length=500)


class PendingActionView(BaseModel):
    tool: str
    args: Any
    reason: str


class RunView(BaseModel):
    id: str
    status: RunStatus
    question: str
    answer: str | None
    citations: list[Citation]
    warnings: list[str]
    pending_action: PendingActionView | None
    error: RunError | None
    trace: list[TraceStep]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, run: AgentRun) -> RunView:
        pending = run.pending_action
        return cls(
            id=run.id,
            status=run.status,
            question=run.question,
            answer=run.answer,
            citations=run.citations,
            warnings=run.warnings,
            pending_action=PendingActionView(
                tool=pending.tool_call.name, args=pending.tool_call.args, reason=pending.reason
            )
            if pending
            else None,
            error=run.error,
            trace=run.trace,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class DocumentView(BaseModel):
    id: str
    title: str
    mime_type: str
    created_at: datetime
    chunk_count: int
    stale: bool

    @classmethod
    def of(cls, doc: Document, stale: bool) -> DocumentView:
        return cls(
            id=doc.id,
            title=doc.title,
            mime_type=doc.mime_type,
            created_at=doc.created_at,
            chunk_count=doc.chunk_count,
            stale=stale,
        )
