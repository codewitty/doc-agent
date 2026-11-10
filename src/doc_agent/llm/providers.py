"""Chat models built from a key the user brings with their request.

The key lives for one request: it is passed to the SDK client and dropped.
Embeddings stay on the server's embedder, so search works the same whichever
model answers (Anthropic has no embeddings API, and vectors from different
embedders cannot be compared).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from doc_agent.llm.anthropic_adapter import (
    DEFAULT_CLAUDE_MODEL,
    AnthropicChatModel,
    workspace_error,
)
from doc_agent.llm.openai_adapter import DEFAULT_CHAT_MODEL, OpenAIChatModel
from doc_agent.ports import ChatModel, ModelAuthError

Provider = Literal["openai", "anthropic"]
PROVIDERS: tuple[str, ...] = get_args(Provider)
DEFAULT_MODELS: dict[str, str] = {"openai": DEFAULT_CHAT_MODEL, "anthropic": DEFAULT_CLAUDE_MODEL}


@dataclass(frozen=True)
class ModelKey:
    """A user's key for one request. The repr leaves the key out so it can't reach a log."""

    provider: Provider
    key: str = field(repr=False)
    # Anthropic keys that aren't tied to a workspace must name one on every request
    workspace: str | None = None


def _anthropic(k: ModelKey, timeout: float) -> AsyncAnthropic:
    headers = {"anthropic-workspace-id": k.workspace} if k.workspace else None
    return AsyncAnthropic(api_key=k.key, max_retries=0, timeout=timeout, default_headers=headers)


def chat_model_for(k: ModelKey, model: str | None = None) -> ChatModel:
    # SDK retries are off: the agent retries once itself and records each attempt
    if k.provider == "openai":
        return OpenAIChatModel(
            AsyncOpenAI(api_key=k.key, max_retries=0, timeout=30), model or DEFAULT_CHAT_MODEL
        )
    return AnthropicChatModel(_anthropic(k, 60), model or DEFAULT_CLAUDE_MODEL)


async def verify_key(k: ModelKey) -> None:
    """Lists models, which costs nothing, to check a key before it is used."""
    try:
        if k.provider == "openai":
            await AsyncOpenAI(api_key=k.key, max_retries=0, timeout=15).models.list()
        else:
            await _anthropic(k, 15).models.list(limit=1)
    except anthropic.BadRequestError as e:
        raise ModelAuthError(workspace_error(e) or "anthropic could not use this key") from e
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
    ) as e:
        raise ModelAuthError(f"{k.provider} rejected the API key") from e
