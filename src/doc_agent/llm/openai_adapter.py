from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAI

from doc_agent.domain import Message, ToolCall
from doc_agent.ports import FinalAnswer, ModelTurn, ToolCalls, ToolSpec

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def _parse_args(raw: str) -> Any:
    # Malformed JSON is passed through as a string so the tool layer reports
    # invalid_arguments back to the model instead of crashing the run
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class OpenAIChatModel:
    def __init__(self, client: AsyncOpenAI, model: str = DEFAULT_CHAT_MODEL) -> None:
        self.client = client
        self.name = f"openai:{model}"
        self._model = model

    async def next(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        res = await self.client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=to_openai_messages(messages),  # type: ignore[arg-type]
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ],
        )
        if not res.choices:
            raise RuntimeError("model returned no choices")
        msg = res.choices[0].message
        if msg.tool_calls:
            return ToolCalls(
                content=msg.content,
                calls=[
                    ToolCall(id=c.id, name=c.function.name, args=_parse_args(c.function.arguments))
                    for c in msg.tool_calls
                    if c.type == "function"
                ],
            )
        return FinalAnswer(content=msg.content or "")


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.client = client
        self.name = f"openai:{model}"
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in sorted(res.data, key=lambda d: d.index)]
