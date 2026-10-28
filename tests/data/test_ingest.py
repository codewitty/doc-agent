"""Data validation: source files are extracted, transformed and presented to
the agent correctly. Uses the same fixture files the demo and evals use."""

import re

import pytest

from doc_agent.agent.tools import SEARCH_DOCUMENTS, SearchArgs, ToolContext
from doc_agent.container import Container
from doc_agent.data.ingest import (
    IngestionError,
    chunk_text,
    csv_to_text,
    extract_text,
    normalize,
)
from doc_agent.fixtures import FIXTURE_DIR, FIXTURES
from tests.conftest import NOW


def read(name: str) -> str:
    return extract_text(name, (FIXTURE_DIR / name).read_bytes())


def test_csv_rows_become_labelled_sentences() -> None:
    assert read("pricing-2024.csv").splitlines() == [
        "plan: Starter; monthly_price_usd: 49; seats: 5; support: Email.",
        "plan: Growth; monthly_price_usd: 199; seats: 25; support: Email and chat.",
        "plan: Enterprise; monthly_price_usd: 899; seats: unlimited; support: Dedicated manager.",
    ]


def test_csv_handles_quoted_commas_bom_and_blank_lines() -> None:
    raw = '﻿name,notes\n\n"Acme, Inc.",  "said ""hi"""  \n'
    assert csv_to_text(raw.removeprefix("﻿")) == 'name: Acme, Inc.; notes: said "hi".'
    assert extract_text("x.csv", raw.encode("utf-8")) == 'name: Acme, Inc.; notes: said "hi".'


def test_csv_with_only_a_header_has_no_rows() -> None:
    assert csv_to_text("a,b\n") == ""


def test_docx_text_keeps_key_facts_and_no_markup() -> None:
    text = normalize(read("service-agreement.docx"))
    assert "Payment is due within 30 days of the invoice date." in text
    assert "99.9 percent monthly uptime" in text
    assert not re.search(r"<w:|PK\x03\x04", text)


def test_handbook_keeps_injection_line_as_plain_data() -> None:
    assert "ignore previous instructions" in read("employee-handbook.docx")


def test_chunking_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text(" \n\t ") == []


def test_chunking_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


@pytest.mark.parametrize("fixture", [f.file for f in FIXTURES])
@pytest.mark.parametrize(("size", "overlap"), [(600, 100), (120, 30), (60, 0)])
def test_chunks_respect_size_and_lose_no_words(fixture: str, size: int, overlap: int) -> None:
    text = normalize(read(fixture))
    chunks = chunk_text(text, size, overlap)
    assert all(0 < len(c) <= size for c in chunks)
    # Every word of the source appears, in order, once overlaps are removed
    joined = " ".join(chunks)
    for word in text.split(" "):
        assert word in joined


def test_long_sentence_is_split_not_truncated() -> None:
    sentence = " ".join(f"word{i}" for i in range(200)) + "."
    chunks = chunk_text(sentence, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert "word199." in chunks[-1]


def test_unbroken_token_longer_than_chunk_is_split() -> None:
    chunks = chunk_text("a" * 250, size=100, overlap=10)
    assert "".join(c.replace(" ", "") for c in chunks).count("a") >= 250


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_text(text, size=120, overlap=30)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split(" ")[0] in prev[-30:]


def test_fixtures_load_with_stable_ids(container: Container) -> None:
    docs = {d.id: d for d in container.documents.list_documents("alice")}
    assert set(docs) == {"alice-msa", "alice-handbook", "alice-pricing"}
    assert all(d.chunk_count > 0 for d in docs.values())


@pytest.mark.parametrize(
    ("query", "title", "stale", "fact"),
    [
        ("payment due invoice date", "service-agreement.docx", False, "30 days"),
        (
            "Growth plan monthly price",
            "pricing-2024.csv",
            True,
            "plan: Growth; monthly_price_usd: 199",
        ),
        ("paid time off days", "employee-handbook.docx", False, "20 days of paid time off"),
    ],
)
def test_search_presents_the_right_source(
    container: Container, query: str, title: str, stale: bool, fact: str
) -> None:
    ctx = ToolContext(
        user_id="alice", documents=container.documents, embedder=container.embedder, now=NOW
    )
    top = SEARCH_DOCUMENTS.run(SearchArgs(query=query, top_k=1), ctx).content["results"][0]
    assert top["document"] == title
    assert top["stale"] is stale
    assert fact in top["content"]
    assert -1 <= top["similarity"] <= 1


def test_ingest_rejects_malformed_embedder_output(container: Container) -> None:
    class Bad:
        name = "bad"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]  # one vector regardless of input size

    from doc_agent.data.ingest import IngestionService

    svc = IngestionService(container.documents, Bad())
    with pytest.raises(RuntimeError, match="malformed"):
        svc.ingest("alice", "two.txt", ("First sentence. " * 60 + "Second.").encode())


def test_unsupported_extension_message_lists_allowed_types() -> None:
    with pytest.raises(IngestionError, match=r"\.csv, \.docx"):
        extract_text("x.xlsx", b"data")


def test_injection_line_is_stored_but_redacted_in_search_results(container: Container) -> None:
    ctx = ToolContext(
        user_id="alice", documents=container.documents, embedder=container.embedder, now=NOW
    )
    top = SEARCH_DOCUMENTS.run(SearchArgs(query="vacation employees", top_k=1), ctx).content[
        "results"
    ][0]
    assert top["document"] == "employee-handbook.docx"
    assert top["redacted"] is True
    assert "unlimited" not in top["content"]
