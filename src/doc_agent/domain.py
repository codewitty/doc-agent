"""Core types shared by the agent, data layer and API.

An AgentRun holds everything needed to resume: messages, trace, retrieved
chunks and any pending action. It is saved after every step so a run can
pause for operator approval and finish in a later request.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED = "refused"


class ToolCall(BaseModel):
    id: str
    name: str
    args: Any


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


StepKind = Literal[
    "input_check",
    "model",
    "tool",
    "approval_requested",
    "approval_decision",
    "output_check",
    "error",
]


class TraceStep(BaseModel):
    index: int
    kind: StepKind
    at: datetime
    duration_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str


class PendingAction(BaseModel):
    tool_call: ToolCall
    reason: str


ErrorCode = Literal["model_unavailable", "step_limit"]


class RunError(BaseModel):
    code: ErrorCode
    message: str


class AgentRun(BaseModel):
    id: str
    user_id: str
    question: str
    status: RunStatus
    messages: list[Message] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    # chunk id -> citation, for every chunk a tool actually returned
    retrieved: dict[str, Citation] = Field(default_factory=dict)
    pending_action: PendingAction | None = None
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: RunError | None = None
    model_calls: int = 0
    created_at: datetime
    updated_at: datetime


class Document(BaseModel):
    id: str
    user_id: str
    title: str
    mime_type: str
    created_at: datetime
    chunk_count: int = 0


class NewChunk(BaseModel):
    index: int
    content: str
    embedding: list[float]


class ChunkHit(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    document_created_at: datetime
    content: str
    similarity: float = Field(ge=-1.0, le=1.0)
