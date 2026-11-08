"""Failure injection by wrapping real adapters, so the agent code under test
is exactly what runs in production."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from doc_agent.domain import Message
from doc_agent.ports import ChatModel, DocumentRepository, Embedder, ModelTurn, ToolCalls, ToolSpec
from evals.cases import Injection


class SlowSearch:
    def __init__(self, inner: DocumentRepository, delay_s: float) -> None:
        self.inner, self.delay_s = inner, delay_s

    def search(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(self.delay_s)
        return self.inner.search(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class MalformedSearch:
    def __init__(self, inner: DocumentRepository) -> None:
        self.inner = inner

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return [{"chunk_id": "", "document_id": None, "similarity": 42}]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class DownEmbedder:
    name: str = "down"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("embedding service unavailable (injected)")


class DownModel:
    name = "down"

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        raise ConnectionError("model provider unavailable (injected)")


class CorruptFirstToolCall:
    """Replaces the arguments of the model's first tool call with broken JSON,
    as a provider returning malformed tool-call arguments would."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.name = f"{inner.name}+corrupt"
        self._done = False

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        turn = await self.inner.next(messages, tools)
        if not self._done and isinstance(turn, ToolCalls) and turn.calls:
            self._done = True
            first = turn.calls[0].model_copy(update={"args": '{"query": '})
            return ToolCalls(calls=[first, *turn.calls[1:]], content=turn.content)
        return turn


def apply(
    injection: Injection | None,
    model: ChatModel,
    documents: DocumentRepository,
    embedder: Embedder,
    tool_timeout_s: float,
) -> tuple[ChatModel, DocumentRepository, Embedder]:
    match injection:
        case None:
            return model, documents, embedder
        case "search_timeout":
            return model, SlowSearch(documents, tool_timeout_s * 3), embedder
        case "malformed_search":
            return model, MalformedSearch(documents), embedder
        case "embedder_down":
            return model, documents, DownEmbedder()
        case "model_down":
            return DownModel(), documents, embedder
        case "corrupt_tool_args":
            return CorruptFirstToolCall(model), documents, embedder
