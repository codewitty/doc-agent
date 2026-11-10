from __future__ import annotations

import json
from typing import Any, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

from doc_agent.domain import Message, ToolCall
from doc_agent.ports import FinalAnswer, ModelAuthError, ModelTurn, ToolCalls, ToolSpec

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
REFUSAL_TEXT = "I can't help with that request."


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Splits out the system prompt and converts the rest to Messages API turns."""
    system = "\n\n".join(m.content or "" for m in messages if m.role == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content or "",
                "is_error": _is_error(m.content),
            }
            # All results for one assistant turn go back in a single user message
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            out.append({"role": "assistant", "content": m.provider_content or _blocks(m)})
        else:
            out.append({"role": "user", "content": m.content or ""})
    return system, out


def _blocks(m: Message) -> list[dict[str, Any]]:
    """Rebuilds an assistant turn that another provider produced."""
    blocks: list[dict[str, Any]] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    for c in m.tool_calls:
        args = c.args if isinstance(c.args, dict) else {}
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": args})
    return blocks


def _is_error(content: str | None) -> bool:
    try:
        body = json.loads(content or "")
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and "error" in body


class AnthropicChatModel:
    def __init__(self, client: AsyncAnthropic, model: str = DEFAULT_CLAUDE_MODEL) -> None:
        self.client = client
        self.name = f"anthropic:{model}"
        self._model = model

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        system, turns = to_anthropic_messages(messages)
        tool_params: list[ToolParam] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        try:
            res = await self.client.messages.create(
                model=self._model,
                max_tokens=16000,
                system=system,
                messages=cast(list[MessageParam], turns),
                tools=tool_params,
                # Document Q&A needs little deliberation; medium is also this model's default
                output_config={"effort": "medium"},
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ModelAuthError("Anthropic rejected the API key") from e

        if res.stop_reason == "refusal":
            return FinalAnswer(content=REFUSAL_TEXT)
        if res.stop_reason == "max_tokens":
            raise RuntimeError("model hit max_tokens before finishing its turn")

        # Kept verbatim: thinking blocks are only valid if replayed unchanged
        raw = [b.model_dump(mode="json", exclude_none=True) for b in res.content]
        text = "".join(b.text for b in res.content if b.type == "text") or None
        calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in res.content
            if b.type == "tool_use"
        ]
        if calls:
            return ToolCalls(calls=calls, content=text, raw=raw)
        return FinalAnswer(content=text or "", raw=raw)
