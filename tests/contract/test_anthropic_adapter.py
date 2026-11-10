"""The Claude adapter against the real Anthropic SDK with a mocked HTTP transport.

Checks the request bodies we send and the responses we parse. The part most
likely to break silently is thinking-block replay: Claude rejects a history
whose thinking blocks were edited or dropped, so these tests check that each
assistant turn goes back byte for byte, including across an approval pause
that reloads the run from the database.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from doc_agent.agent.orchestrator import Agent
from doc_agent.agent.tools import DEFAULT_TOOLS
from doc_agent.container import Container
from doc_agent.domain import Message, RunStatus, ToolCall
from doc_agent.llm.anthropic_adapter import REFUSAL_TEXT, AnthropicChatModel
from doc_agent.ports import FinalAnswer, ToolCalls
from tests.conftest import NOW

pytestmark = pytest.mark.anyio

Responder = Callable[[dict[str, Any]], httpx2.Response]
SPECS = [t.spec for t in DEFAULT_TOOLS]
KEY = "sk-ant-test-secret-key"


def message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def thinking(sig: str) -> dict[str, Any]:
    # Opus 5.5 returns thinking with empty text by default; the signature is what matters
    return {"type": "thinking", "thinking": "", "signature": sig}


def tool_use(name: str, args: dict[str, Any], id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": id, "name": name, "input": args}


class Recorder:
    def __init__(self, responders: list[Responder]) -> None:
        self.responders = list(responders)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, "headers": request.headers, "body": body})
        return self.responders.pop(0)(body)


def claude(rec: Recorder) -> AnthropicChatModel:
    client = AsyncAnthropic(
        api_key=KEY,
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(rec)),
    )
    return AnthropicChatModel(client)


def agent(container: Container, model: AnthropicChatModel) -> Agent:
    return Agent(
        model=model,
        documents=container.documents,
        runs=container.runs,
        embedder=container.embedder,
        now=lambda: NOW,
    )


async def test_request_shape() -> None:
    rec = Recorder([lambda _: message([{"type": "text", "text": "hi"}])])
    turn = await claude(rec).next(
        [Message(role="system", content="sys"), Message(role="user", content="q")], SPECS
    )
    assert isinstance(turn, FinalAnswer) and turn.content == "hi"

    req = rec.requests[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["authorization"] == f"Bearer {KEY}"
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert body["output_config"] == {"effort": "medium"}
    # Sampling parameters are rejected by current models
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    names = [t["name"] for t in body["tools"]]
    assert names == ["search_documents", "list_documents", "delete_document"]
    assert body["tools"][0]["input_schema"]["additionalProperties"] is False


async def test_parses_tool_use_and_keeps_raw_content() -> None:
    content = [thinking("sig-a"), tool_use("search_documents", {"query": "x"})]
    rec = Recorder([lambda _: message(content, "tool_use")])
    turn = await claude(rec).next([Message(role="user", content="q")], SPECS)
    assert isinstance(turn, ToolCalls)
    assert turn.calls == [ToolCall(id="toolu_1", name="search_documents", args={"query": "x"})]
    assert turn.raw == content


async def test_parallel_tool_results_go_back_in_one_user_message() -> None:
    rec = Recorder([lambda _: message([{"type": "text", "text": "done"}])])
    history = [
        Message(role="user", content="q"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="t1", name="search_documents", args={"query": "a"}),
                ToolCall(id="t2", name="list_documents", args={}),
            ],
        ),
        Message(role="tool", tool_call_id="t1", content='{"results": []}'),
        Message(role="tool", tool_call_id="t2", content='{"error": "tool_timeout"}'),
    ]
    await claude(rec).next(history, SPECS)
    _, assistant, results = rec.requests[0]["body"]["messages"]
    # A turn from another provider is rebuilt as text + tool_use blocks
    assert [b["type"] for b in assistant["content"]] == ["tool_use", "tool_use"]
    assert results == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": '{"results": []}',
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "t2",
                "content": '{"error": "tool_timeout"}',
                "is_error": True,
            },
        ],
    }


async def test_full_run_replays_thinking_blocks_verbatim(container: Container) -> None:
    first = [thinking("sig-1"), tool_use("search_documents", {"query": "payment due invoice"})]

    def cite(body: dict[str, Any]) -> httpx2.Response:
        result = json.loads(body["messages"][-1]["content"][0]["content"])["results"][0]
        return message(
            [thinking("sig-2"), {"type": "text", "text": f"30 days [chunk:{result['chunk_id']}]"}]
        )

    rec = Recorder([lambda _: message(first, "tool_use"), cite])
    run = await agent(container, claude(rec)).start("alice", "When is payment due?")

    assert run.status == RunStatus.COMPLETED
    assert [c.document_title for c in run.citations] == ["service-agreement.docx"]
    assert rec.requests[1]["body"]["messages"][1] == {"role": "assistant", "content": first}


async def test_approval_resume_replays_history_unchanged(container: Container) -> None:
    first = [thinking("sig-1"), tool_use("delete_document", {"document_id": "alice-pricing"})]
    rec = Recorder(
        [
            lambda _: message(first, "tool_use"),
            lambda _: message([{"type": "text", "text": "Deleted."}]),
        ]
    )
    paused = await agent(container, claude(rec)).start("alice", "Delete the pricing document")
    assert paused.status == RunStatus.AWAITING_APPROVAL

    # A fresh agent and client, as after a server restart: state comes from the database
    resumed = await agent(container, claude(rec)).decide("alice", paused.id, approve=True)
    assert resumed is not None and resumed.status == RunStatus.COMPLETED

    sent = rec.requests[1]["body"]["messages"]
    assert sent[0] == rec.requests[0]["body"]["messages"][0]
    assert sent[1] == {"role": "assistant", "content": first}
    assert sent[2]["content"][0]["tool_use_id"] == "toolu_1"


async def test_refusal_becomes_a_plain_answer() -> None:
    rec = Recorder([lambda _: message([], "refusal")])
    turn = await claude(rec).next([Message(role="user", content="q")], SPECS)
    assert turn == FinalAnswer(REFUSAL_TEXT)


async def test_rejected_key_fails_fast_without_leaking_it(container: Container) -> None:
    error = httpx2.Response(
        401,
        json={
            "type": "error",
            "error": {"type": "authentication_error", "message": f"invalid x-api-key {KEY}"},
        },
    )
    rec = Recorder([lambda _: error, lambda _: error])
    run = await agent(container, claude(rec)).start("alice", "When is payment due?")

    assert run.status == RunStatus.FAILED
    assert run.error is not None and run.error.code == "model_auth_failed"
    assert len(rec.requests) == 1  # a bad key fails the same way on retry
    assert KEY not in run.model_dump_json()


@pytest.mark.parametrize("status", [429, 500, 529])
async def test_provider_errors_fail_the_run_visibly(container: Container, status: int) -> None:
    error = httpx2.Response(
        status, json={"type": "error", "error": {"type": "api_error", "message": "nope"}}
    )
    rec = Recorder([lambda _: error, lambda _: error])
    run = await agent(container, claude(rec)).start("alice", "When is payment due?")
    assert run.status == RunStatus.FAILED
    assert run.error is not None and run.error.code == "model_unavailable"
    assert len(rec.requests) == 2  # one retry, then give up
