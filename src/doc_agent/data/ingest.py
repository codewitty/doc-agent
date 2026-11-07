"""Turns uploaded files into embedded chunks.

extract -> normalize -> chunk -> embed -> store. Each stage validates its
output so a bad file fails at upload with a clear error instead of producing
empty or garbled context for the agent later.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime
from pathlib import PurePath
from zipfile import BadZipFile

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from doc_agent.domain import Document, NewChunk
from doc_agent.ports import DocumentRepository, Embedder

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class IngestionError(ValueError):
    """The file cannot be turned into usable text. Safe to show to the user."""


def extract_text(filename: str, data: bytes) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in MIME_TYPES:
        allowed = ", ".join(sorted(MIME_TYPES))
        raise IngestionError(f"Unsupported file type {suffix or '(none)'}. Allowed: {allowed}")
    try:
        if suffix == ".docx":
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        text = data.decode("utf-8-sig")
        if suffix == ".csv":
            return csv_to_text(text)
        return text
    except IngestionError:
        raise
    except UnicodeDecodeError as e:
        raise IngestionError("File is not valid UTF-8 text") from e
    except (PdfReadError, BadZipFile, KeyError, ValueError) as e:
        # python-docx raises BadZipFile/KeyError/ValueError on corrupt input
        raise IngestionError(f"Could not read {suffix} file: it may be corrupt") from e


def csv_to_text(text: str) -> str:
    """One sentence per row, each value labelled with its header.

    Rows survive chunking as self-contained facts ("plan: Growth; seats: 25.")
    instead of bare values whose meaning lives in a header that may land in a
    different chunk.
    """
    rows = [
        r for r in csv.reader(io.StringIO(text), skipinitialspace=True) if any(c.strip() for c in r)
    ]
    if not rows:
        return ""
    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    lines = []
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise IngestionError(f"CSV row {n} has {len(row)} columns, expected {len(header)}")
        lines.append("; ".join(f"{h}: {v.strip()}" for h, v in zip(header, row, strict=True)) + ".")
    return "\n".join(lines)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack sentences into chunks of at most `size` chars.

    Consecutive chunks share up to `overlap` trailing chars so a fact split
    across a boundary is still retrievable. Sentences longer than `size` are
    split on word boundaries rather than truncated, so no text is dropped.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    normalized = normalize(text)
    if not normalized:
        return []

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(normalized):
        pieces.extend(_split_long(sentence, size - overlap - 1))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail} {piece}" if tail else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    out: list[str] = []
    current = ""
    for word in sentence.split(" "):
        while len(word) > limit:
            if current:
                out.append(current)
                current = ""
            out.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` chars of text, starting on a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and text[-overlap - 1] != " ":
        tail = tail.split(" ", 1)[1] if " " in tail else ""
    return tail


class IngestionService:
    def __init__(self, repo: DocumentRepository, embedder: Embedder) -> None:
        self.repo = repo
        self.embedder = embedder

    def ingest(
        self,
        user_id: str,
        filename: str,
        data: bytes,
        created_at: datetime | None = None,
        document_id: str | None = None,
    ) -> Document:
        if not data:
            raise IngestionError("File is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise IngestionError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

        chunks = chunk_text(extract_text(filename, data))
        if not chunks:
            raise IngestionError("No text could be extracted from the file")

        vectors = self.embedder.embed(chunks)
        if len(vectors) != len(chunks) or len({len(v) for v in vectors}) != 1:
            raise RuntimeError("Embedder returned a malformed batch")

        title = PurePath(filename).name
        return self.repo.add_document(
            user_id=user_id,
            title=title,
            mime_type=MIME_TYPES[PurePath(filename).suffix.lower()],
            chunks=[
                NewChunk(index=i, content=c, embedding=v)
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
            ],
            created_at=created_at or datetime.now(UTC),
            document_id=document_id,
        )
