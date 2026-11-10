"""Bring-your-own-key: a user's OpenAI or Anthropic key arrives in headers on
each request, picks the model for that request, and is never stored."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from doc_agent.container import Container
from doc_agent.llm.providers import Provider
from doc_agent.ports import ChatModel, ModelAuthError
from tests.conftest import ALICE, ScriptedModel, auth, call, final

KEY = "sk-ant-user-secret-123"


def keyed(provider: str = "anthropic", key: str = KEY) -> dict[str, str]:
    return {**auth(ALICE), "X-Model-Provider": provider, "X-Model-Key": key}


@pytest.fixture
def built(container: Container) -> list[tuple[str, str]]:
    """Records which provider and key each request built a model with."""
    seen: list[tuple[str, str]] = []
    model = ScriptedModel(
        [
            call("delete_document", {"document_id": "alice-pricing"}),
            final("Deleted alice-pricing."),
        ]
    )

    def factory(provider: Provider, key: str) -> ChatModel:
        seen.append((provider, key))
        return model

    container.chat_model_for = factory
    return seen


def start(client: TestClient, headers: dict[str, str], q: str = "Delete pricing") -> Any:
    return client.post("/api/runs", json={"question": q}, headers=headers)


def test_without_a_key_the_server_model_answers(
    client: TestClient, built: list[tuple[str, str]]
) -> None:
    assert start(client, auth(ALICE), "When is payment due on invoices?").status_code == 201
    assert built == []


def test_user_key_picks_the_model_for_start_and_decision(
    client: TestClient, built: list[tuple[str, str]]
) -> None:
    run = start(client, keyed()).json()
    assert run["status"] == "awaiting_approval"
    res = client.post(f"/api/runs/{run['id']}/decision", json={"approve": True}, headers=keyed())
    assert res.json()["status"] == "completed"
    assert built == [("anthropic", KEY), ("anthropic", KEY)]


def test_key_is_not_stored_returned_or_logged(
    client: TestClient,
    container: Container,
    built: list[tuple[str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        res = start(client, keyed())
    run_id = res.json()["id"]
    stored = container.runs.get("alice", run_id)
    assert stored is not None
    assert KEY not in res.text
    assert KEY not in stored.model_dump_json()
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({"X-Model-Provider": "gemini", "X-Model-Key": "k"}, "must be one of"),
        ({"X-Model-Provider": "openai"}, "X-Model-Key is required"),
        ({"X-Model-Provider": "openai", "X-Model-Key": "  "}, "X-Model-Key is required"),
        ({"X-Model-Key": "k"}, "must be one of"),
    ],
)
def test_malformed_key_headers_are_rejected(
    client: TestClient, built: list[tuple[str, str]], headers: dict[str, str], detail: str
) -> None:
    res = start(client, {**auth(ALICE), **headers})
    assert res.status_code == 400
    assert detail in res.json()["detail"]
    assert built == []


def test_check_accepts_a_working_key(client: TestClient, container: Container) -> None:
    checked: list[tuple[str, str]] = []

    async def verify(provider: Provider, key: str) -> None:
        checked.append((provider, key))

    container.verify_key = verify
    res = client.post("/api/model/check", headers=keyed("openai", "sk-good"))
    assert res.status_code == 200
    assert res.json() == {"provider": "openai", "model": "gpt-4o-mini", "status": "ok"}
    assert checked == [("openai", "sk-good")]


def test_check_reports_a_rejected_key(client: TestClient, container: Container) -> None:
    async def verify(provider: Provider, key: str) -> None:
        raise ModelAuthError("anthropic rejected the API key")

    container.verify_key = verify
    res = client.post("/api/model/check", headers=keyed())
    assert res.status_code == 400
    assert res.json()["detail"] == "anthropic rejected the API key"


def test_check_needs_auth_and_a_key(client: TestClient) -> None:
    no_token = {"X-Model-Provider": "openai", "X-Model-Key": "k"}
    assert client.post("/api/model/check", headers=no_token).status_code == 401
    assert client.post("/api/model/check", headers=auth(ALICE)).status_code == 400
