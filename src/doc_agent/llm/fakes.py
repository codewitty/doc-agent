"""Deterministic stand-ins for the model and embedder.

They back the keyless demo, the browser test, and the baseline eval run.
HeuristicChatModel follows the same tool protocol a real model would, so the
orchestrator, API and UI run the real code paths around it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from doc_agent.agent.guardrails import NO_INFO_ANSWER
from doc_agent.domain import Message, ToolCall
from doc_agent.ports import FinalAnswer, ModelTurn, ToolCalls, ToolSpec

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    # Crude plural folding so "invoices" matches "invoice"
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


class HashingEmbedder:
    """Bag-of-words hashed into a fixed number of dimensions, L2-normalized."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims
        self.name = f"hashing:{dims}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in tokenize(text):
            h = int(hashlib.md5(_stem(word).encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


RELEVANCE_THRESHOLD = 0.2


class HeuristicChatModel:
    name = "heuristic"

    def __init__(self) -> None:
        self._seq = 0

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        question = next((m.content or "" for m in reversed(messages) if m.role == "user"), "")
        tool_msgs = [m for m in messages if m.role == "tool"]
        wants_delete = re.search(r"\b(delete|remove)\b", question, re.I) is not None

        if not tool_msgs:
            if wants_delete:
                return self._call("list_documents", {})
            return self._call("search_documents", {"query": question, "top_k": 3})

        last = tool_msgs[-1]
        calls = [c for m in messages if m.role == "assistant" for c in m.tool_calls]
        last_call = next((c for c in calls if c.id == last.tool_call_id), None)
        result: dict[str, Any] = json.loads(last.content or "{}")

        if result.get("error") == "rejected_by_operator":
            return FinalAnswer("The operator rejected that action, so nothing was changed.")
        if result.get("error"):
            return FinalAnswer("I couldn't complete the lookup right now. Please try again later.")

        match last_call.name if last_call else None:
            case "list_documents":
                q = question.lower()
                doc = next(
                    (d for d in result["documents"] if d["title"].lower().rsplit(".", 1)[0] in q),
                    None,
                )
                if doc is None:
                    return FinalAnswer("I couldn't find a document with that name.")
                return self._call("delete_document", {"document_id": doc["id"]})
            case "delete_document":
                return FinalAnswer(f"Deleted document {result['deleted']}.")
            case "search_documents":
                top = (result.get("results") or [None])[0]
                if top is None or top["similarity"] < RELEVANCE_THRESHOLD:
                    return FinalAnswer(NO_INFO_ANSWER)
                stale = " Note: this source may be out of date." if top["stale"] else ""
                return FinalAnswer(
                    f"{_best_sentence(top['content'], question)} [chunk:{top['chunk_id']}]{stale}"
                )
        return FinalAnswer(NO_INFO_ANSWER)

    def _call(self, name: str, args: dict[str, Any]) -> ToolCalls:
        self._seq += 1
        return ToolCalls(calls=[ToolCall(id=f"call_{self._seq}", name=name, args=args)])


def _best_sentence(content: str, question: str) -> str:
    terms = {_stem(t) for t in tokenize(question)}
    sentences = re.split(r"(?<=[.!?])\s+", content)
    return max(sentences, key=lambda s: len(terms & {_stem(t) for t in tokenize(s)}))
