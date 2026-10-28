from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from doc_agent.agent.orchestrator import Agent, Limits
from doc_agent.api.app import create_app
from doc_agent.config import Settings
from doc_agent.container import Container, build_container
from doc_agent.domain import Message, ToolCall
from doc_agent.fixtures import seed_fixtures
from doc_agent.ports import ChatModel, FinalAnswer, ModelTurn, ToolCalls, ToolSpec

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
ALICE = "alice-dev-token"
BOB = "bob-dev-token"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


Step = ModelTurn | Callable[[list[Message]], ModelTurn | Awaitable[ModelTurn]]


class ScriptedModel:
    """Plays back a fixed list of turns. A callable step can inspect the
    messages so far, raise, or hang to simulate model failures."""

    name = "scripted"

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list[Message]] = []

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        self.calls.append([m.model_copy(deep=True) for m in messages])
        if not self.steps:
            raise AssertionError("ScriptedModel ran out of steps")
        step = self.steps.pop(0)
        if callable(step):
            result = step(messages)
            return await result if inspect.isawaitable(result) else result
        return step


def call(name: str, args: Any, id: str | None = None) -> ToolCalls:
    return ToolCalls(calls=[ToolCall(id=id or f"call_{name}", name=name, args=args)])


def final(content: str) -> FinalAnswer:
    return FinalAnswer(content)


def last_tool_result(messages: list[Message]) -> dict[str, Any]:
    m = next(m for m in reversed(messages) if m.role == "tool")
    result: dict[str, Any] = json.loads(m.content or "{}")
    return result


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(env="test", database_url=f"sqlite:///{tmp_path}/test.db", seed_users=("alice",))


@pytest.fixture
def container(settings: Settings) -> Container:
    c = build_container(settings)
    seed_fixtures(c.ingestion, "alice")
    seed_fixtures(c.ingestion, "bob")
    return c


@pytest.fixture
def make_agent(container: Container) -> Callable[..., Agent]:
    def make(model: ChatModel | None = None, **limits: Any) -> Agent:
        ids = iter(f"run-{i}" for i in range(1, 1000))
        return Agent(
            model=model or container.model,
            documents=container.documents,
            runs=container.runs,
            embedder=container.embedder,
            limits=Limits(**limits),
            now=lambda: NOW,
            new_id=lambda: next(ids),
        )

    return make


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    app = create_app(container, seed=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
