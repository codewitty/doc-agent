from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

DATASET = Path(__file__).with_name("dataset.jsonl")

Injection = Literal[
    "search_timeout", "embedder_down", "malformed_search", "corrupt_tool_args", "model_down"
]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: str
    question: str
    expect_status: Literal["completed", "refused", "failed"] = "completed"
    # Task completion: every phrase must appear (case-insensitive)
    must_include: list[str] = []
    # Safety: none of these may appear, e.g. injected or fabricated facts
    must_not_include: list[str] = []
    expect_no_info: bool = False
    # Relevance and groundedness: titles that must be retrieved first and cited
    expect_sources: list[str] = []
    # Tool accuracy
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    # Operator control
    approval: Literal["approve", "reject"] | None = None
    expect_deleted: str | None = None
    expect_kept: str | None = None
    # Failure handling
    inject: Injection | None = None
    expect_warning: bool = False


def load_cases(path: Path = DATASET) -> list[EvalCase]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    cases = [EvalCase.model_validate_json(line) for line in lines]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in dataset")
    return cases
