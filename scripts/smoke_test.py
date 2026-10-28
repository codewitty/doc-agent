"""Post-deploy check: the site is up, auth works, and one grounded run completes.

SMOKE_BASE_URL=https://... SMOKE_TOKEN=... python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import httpx

base = os.environ["SMOKE_BASE_URL"].rstrip("/")
token = os.environ["SMOKE_TOKEN"]
# Render's free tier sleeps when idle; the first request can take ~60s
client = httpx.Client(base_url=base, timeout=90)
failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


health = client.get("/api/health")
check("health", health.status_code == 200, health.text)
check("rejects missing token", client.get("/api/documents").status_code == 401)

headers = {"Authorization": f"Bearer {token}"}
docs = client.get("/api/documents", headers=headers)
check("lists documents", docs.status_code == 200 and len(docs.json()) > 0, docs.status_code)

run = client.post(
    "/api/runs", json={"question": "When is payment due on invoices?"}, headers=headers
).json()
check("run completes", run.get("status") == "completed", run.get("status"))
check("run is cited", bool(run.get("citations")), run.get("answer"))

sys.exit(1 if failures else 0)
