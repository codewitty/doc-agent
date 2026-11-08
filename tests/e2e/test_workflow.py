"""Critical user workflow in a real browser against a real server process.

Runs in keyless demo mode: heuristic model, hashing embedder, fresh SQLite.
Screenshots of each stage land in reports/screenshots/ for review.
Run with: pytest -m e2e
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

SHOTS = Path("reports/screenshots")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = _free_port()
    db = tmp_path_factory.mktemp("e2e") / "e2e.db"
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db}",
        "DOC_AGENT_MODEL": "heuristic",
        "DOC_AGENT_EMBEDDER": "hashing",
        "DOC_AGENT_SEED_USERS": "alice",
    }
    env.pop("OPENAI_API_KEY", None)
    log = (tmp_path_factory.getbasetemp() / "server.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "doc_agent.main:app", "--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(f"server did not start; see {log.name}")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def test_ask_cite_then_delete_with_approval(server: str, page: Page) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.goto(server)

    page.get_by_label("Access token").fill("wrong")
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_role("alert")).to_contain_text("not accepted")

    page.get_by_label("Access token").fill("alice-dev-token")
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_test_id("who")).to_have_text("Signed in as alice")

    docs = page.get_by_test_id("documents")
    expect(docs.locator("li")).to_have_count(3)
    expect(docs.locator("li", has_text="pricing-2024.csv")).to_contain_text("stale")
    page.screenshot(path=SHOTS / "1-signed-in.png", full_page=True)

    page.get_by_label("Question or request").fill("When is payment due on invoices?")
    page.get_by_role("button", name="Ask").click()
    expect(page.get_by_test_id("run-status")).to_have_attribute("data-status", "completed")
    expect(page.get_by_test_id("answer")).to_contain_text("30 days")
    expect(page.get_by_test_id("citations")).to_contain_text("service-agreement.docx")
    page.get_by_text("Execution trace").click()
    expect(page.get_by_test_id("trace").locator("tr")).to_have_count(5)
    page.screenshot(path=SHOTS / "2-cited-answer.png", full_page=True)

    page.get_by_label("Question or request").fill("Delete the pricing-2024 document")
    page.get_by_role("button", name="Ask").click()
    expect(page.get_by_test_id("run-status")).to_have_attribute("data-status", "awaiting_approval")
    approval = page.get_by_test_id("approval")
    expect(approval).to_contain_text('delete_document({"document_id":"alice-pricing"})')
    # Nothing is deleted before the operator decides
    expect(docs.locator("li")).to_have_count(3)
    page.screenshot(path=SHOTS / "3-awaiting-approval.png", full_page=True)

    page.get_by_role("button", name="Approve").click()
    expect(page.get_by_test_id("run-status")).to_have_attribute("data-status", "completed")
    expect(page.get_by_test_id("answer")).to_contain_text("Deleted document alice-pricing")
    expect(approval).to_be_hidden()
    expect(docs.locator("li")).to_have_count(2)
    expect(docs).not_to_contain_text("pricing-2024.csv")
    page.screenshot(path=SHOTS / "4-approved.png", full_page=True)
