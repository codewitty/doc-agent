"""The keyless demo (heuristic model + hashing embedder) must drive the same
workflow as the real model, end to end over the fixture documents. The
browser test and the baseline eval both depend on this."""

from collections.abc import Callable

import pytest

from doc_agent.agent.guardrails import NO_INFO_ANSWER
from doc_agent.agent.orchestrator import Agent
from doc_agent.container import Container
from doc_agent.domain import RunStatus

pytestmark = pytest.mark.anyio


async def test_grounded_answer_with_citation(make_agent: Callable[..., Agent]) -> None:
    run = await make_agent().start("alice", "When is payment due on invoices?")
    assert run.status == RunStatus.COMPLETED
    assert run.answer is not None and "30 days" in run.answer
    assert run.citations[0].document_title == "service-agreement.docx"
    assert run.warnings == []


async def test_stale_source_is_flagged(make_agent: Callable[..., Agent]) -> None:
    run = await make_agent().start("alice", "What is the Growth plan monthly price?")
    assert run.answer is not None
    assert "199" in run.answer and "out of date" in run.answer


async def test_unanswerable_question_falls_back(make_agent: Callable[..., Agent]) -> None:
    assert (
        await make_agent().start("alice", "Who won the 1998 World Cup?")
    ).answer == NO_INFO_ANSWER


async def test_delete_with_approval(make_agent: Callable[..., Agent], container: Container) -> None:
    agent = make_agent()
    paused = await agent.start("alice", "Delete the pricing-2024 document")
    assert paused.pending_action is not None
    assert paused.pending_action.tool_call.args == {"document_id": "alice-pricing"}

    done = await agent.decide("alice", paused.id, approve=True)
    assert done is not None and done.answer == "Deleted document alice-pricing."
    assert "alice-pricing" not in {d.id for d in container.documents.list_documents("alice")}
