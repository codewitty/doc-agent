from __future__ import annotations

import re
from dataclasses import dataclass

from doc_agent.domain import Citation

NO_INFO_ANSWER = "I'm sorry, I don't have information about that."
REFUSAL_ANSWER = "I can't help with that request."

_UNSAFE = [
    (
        re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I),
        "prompt_injection",
    ),
    (re.compile(r"(reveal|print|show).{0,20}system prompt", re.I), "prompt_injection"),
    (re.compile(r"\b(api[ _-]?key|password|secret token)s?\b", re.I), "credential_request"),
    (re.compile(r"\b(other|another|all) users?'?s? (documents|files|data)", re.I), "cross_tenant"),
]


def check_input(question: str) -> str | None:
    """Return the refusal reason, or None if the question is allowed.

    A pattern list is a cheap first line of defence, not a complete one: the
    system prompt, tenant-scoped tools and the approval gate still apply to
    anything that gets past it.
    """
    for pattern, reason in _UNSAFE:
        if pattern.search(question):
            return reason
    return None


_INSTRUCTION_LIKE = [p for p, reason in _UNSAFE if reason == "prompt_injection"] + [
    re.compile(r"\b(note|message) to (ai|llm|the) (assistants?|models?)\b", re.I),
]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
REDACTED = "[removed: instruction-like text]"


def redact_instructions(content: str) -> tuple[str, bool]:
    """Replace sentences in retrieved text that try to instruct the model.

    Retrieved documents are untrusted. The stored document is not changed;
    only what the model sees for this search result.
    """
    sentences = _SENTENCE.split(content)
    kept = [REDACTED if any(p.search(s) for p in _INSTRUCTION_LIKE) else s for s in sentences]
    return " ".join(kept), kept != sentences


_CITATION = re.compile(r"\[chunk:([^\]\s]+)\]")


@dataclass(frozen=True)
class OutputCheck:
    citations: list[Citation]
    unknown_citations: list[str]
    grounded: bool


def check_output(answer: str, retrieved: dict[str, Citation], searched: bool) -> OutputCheck:
    """Grounded means the fixed no-info reply, or at least one citation of a
    chunk a tool returned in this run. Citing a chunk the run never saw is a
    hallucination and is reported."""
    ids = list(dict.fromkeys(_CITATION.findall(answer)))
    citations = [retrieved[i] for i in ids if i in retrieved]
    unknown = [i for i in ids if i not in retrieved]
    is_no_info = answer.strip() == NO_INFO_ANSWER
    grounded = not unknown and (is_no_info or bool(citations) or not searched)
    return OutputCheck(citations=citations, unknown_citations=unknown, grounded=grounded)
