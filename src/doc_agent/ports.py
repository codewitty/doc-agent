"""Interfaces the agent and API depend on.

Each has a production adapter and a deterministic fake, so tests, CI and
the keyless demo exercise the same code paths as the live site.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from doc_agent.domain import AgentRun, ChunkHit, Document, Message, NewChunk, ToolCall


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    content: str


@dataclass(frozen=True)
class ToolCalls:
    calls: list[ToolCall]
    content: str | None = None


ModelTurn = FinalAnswer | ToolCalls


class ChatModel(Protocol):
    name: str

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn: ...


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DocumentRepository(Protocol):
    def add_document(
        self,
        user_id: str,
        title: str,
        mime_type: str,
        chunks: list[NewChunk],
        created_at: datetime,
        document_id: str | None = None,
    ) -> Document: ...

    def list_documents(self, user_id: str) -> list[Document]: ...

    def search(self, user_id: str, embedding: list[float], top_k: int) -> list[ChunkHit]: ...

    def delete_document(self, user_id: str, document_id: str) -> bool: ...


class RunRepository(Protocol):
    def create(self, run: AgentRun) -> None: ...

    def get(self, user_id: str, run_id: str) -> AgentRun | None: ...

    def save(self, run: AgentRun) -> None: ...
