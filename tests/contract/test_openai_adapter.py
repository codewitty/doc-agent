"""The OpenAI adapters against the real SDK with a mocked HTTP transport.

Checks the request bodies we send and the responses we parse, including a
full multi-turn agent run. It does not prove model quality; the nightly
eval against the live API does that.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from doc_agent.agent.orchestrator import Agent, Limits
from doc_agent.agent.tools import DEFAULT_TOOLS
from doc_agent.container import Container
from doc_agent.domain import Message, RunStatus, ToolCall
from doc_agent.llm.openai_adapter import OpenAIChatModel, OpenAIEmbedder
from doc_agent.ports import FinalAnswer, ToolCalls
from tests.conftest import NOW

pytestmark = pytest.mark.anyio

Responder = Callable[[dict[str, Any]], httpx.Response]


def completion(message: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def tool_call_message(name: str, arguments: str, id: str = "call_1") -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class Recorder:
    def __init__(self, responders: list[Responder]) -> None:
        self.responders = list(responders)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, "body": body})
        return self.responders.pop(0)(body)


def chat_model(recorder: Recorder) -> OpenAIChatModel:
    client = AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )
    return OpenAIChatModel(client)


SPECS = [t.spec for t in DEFAULT_TOOLS]


async def test_request_carries_tools_temperature_and_messages() -> None:
    rec = Recorder([lambda _: completion({"content": "hi"})])
    turn = await chat_model(rec).next(
        [Message(role="system", content="sys"), Message(role="user", content="q")], SPECS
    )
    assert turn == FinalAnswer("hi")

    req = rec.requests[0]
    assert req["path"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    names = [t["function"]["name"] for t in body["tools"]]
    assert names == ["search_documents", "list_documents", "delete_document"]
    search = body["tools"][0]["function"]["parameters"]
    assert search["type"] == "object"
    assert search["required"] == ["query"]
    assert search["additionalProperties"] is False


async def test_parses_tool_calls() -> None:
    rec = Recorder([lambda _: completion(tool_call_message("search_documents", '{"query": "x"}'))])
    turn = await chat_model(rec).next([Message(role="user", content="q")], SPECS)
    assert turn == ToolCalls(
        calls=[ToolCall(id="call_1", name="search_documents", args={"query": "x"})]
    )


async def test_malformed_arguments_pass_through_as_a_string() -> None:
    rec = Recorder([lambda _: completion(tool_call_message("search_documents", '{"query": '))])
    turn = await chat_model(rec).next([Message(role="user", content="q")], SPECS)
    assert isinstance(turn, ToolCalls)
    assert turn.calls[0].args == '{"query": '


async def test_tool_round_trip_is_serialized_in_openai_format() -> None:
    rec = Recorder([lambda _: completion({"content": "done"})])
    history = [
        Message(role="user", content="q"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_9", name="search_documents", args={"query": "x", "top_k": 3})
            ],
        ),
        Message(role="tool", tool_call_id="call_9", content='{"results": []}'),
    ]
    await chat_model(rec).next(history, SPECS)
    _, assistant, tool = rec.requests[0]["body"]["messages"]
    assert assistant == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_9",
                "type": "function",
                "function": {"name": "search_documents", "arguments": '{"query": "x", "top_k": 3}'},
            }
        ],
    }
    assert tool == {"role": "tool", "tool_call_id": "call_9", "content": '{"results": []}'}


async def test_full_agent_run_through_the_adapter(container: Container) -> None:
    def cite_first_result(body: dict[str, Any]) -> httpx.Response:
        result = json.loads(body["messages"][-1]["content"])["results"][0]
        return completion({"content": f"Due in 30 days [chunk:{result['chunk_id']}]"})

    rec = Recorder(
        [
            lambda _: completion(
                tool_call_message("search_documents", '{"query": "payment due invoice"}')
            ),
            cite_first_result,
        ]
    )
    agent = Agent(
        model=chat_model(rec),
        documents=container.documents,
        runs=container.runs,
        embedder=container.embedder,
        now=lambda: NOW,
    )
    run = await agent.start("alice", "When is payment due?")

    assert run.status == RunStatus.COMPLETED
    assert [c.document_title for c in run.citations] == ["service-agreement.docx"]
    assert run.warnings == []
    assert len(rec.requests) == 2
    assert rec.requests[1]["body"]["messages"][-1]["role"] == "tool"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_provider_errors_fail_the_run_visibly(container: Container, status: int) -> None:
    error = httpx.Response(status, json={"error": {"message": "nope", "type": "server_error"}})
    rec = Recorder([lambda _: error, lambda _: error])
    agent = Agent(
        model=chat_model(rec),
        documents=container.documents,
        runs=container.runs,
        embedder=container.embedder,
        limits=Limits(),
        now=lambda: NOW,
    )
    run = await agent.start("alice", "When is payment due?")
    assert run.status == RunStatus.FAILED
    assert run.error is not None and run.error.code == "model_unavailable"
    assert len(rec.requests) == 2  # one retry, then give up


def test_embedder_batches_and_orders_by_index() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["a", "b"]
        # Deliberately out of order
        data = [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": body["model"],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    client = OpenAI(
        api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    assert OpenAIEmbedder(client).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
