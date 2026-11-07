"""Representative documents used by the demo, tests and evals.

Each fixture goes through the real ingestion pipeline. created_at is fixed so
stale-data handling is reproducible: the pricing sheet is older than
STALE_AFTER on purpose, and the handbook contains a prompt-injection line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from doc_agent.data.ingest import IngestionService

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "documents"


@dataclass(frozen=True)
class Fixture:
    slug: str
    file: str
    created_at: datetime


FIXTURES = [
    Fixture("msa", "service-agreement.docx", datetime(2026, 9, 1, tzinfo=UTC)),
    Fixture("handbook", "employee-handbook.docx", datetime(2026, 8, 15, tzinfo=UTC)),
    Fixture("pricing", "pricing-2024.csv", datetime(2024, 11, 1, tzinfo=UTC)),
]


def fixture_document_id(user_id: str, slug: str) -> str:
    return f"{user_id}-{slug}"


def seed_fixtures(ingestion: IngestionService, user_id: str) -> int:
    """Load any fixtures the user is missing. Safe to call on every startup."""
    existing = {d.id for d in ingestion.repo.list_documents(user_id)}
    added = 0
    for f in FIXTURES:
        doc_id = fixture_document_id(user_id, f.slug)
        if doc_id in existing:
            continue
        ingestion.ingest(
            user_id,
            f.file,
            (FIXTURE_DIR / f.file).read_bytes(),
            created_at=f.created_at,
            document_id=doc_id,
        )
        added += 1
    return added
