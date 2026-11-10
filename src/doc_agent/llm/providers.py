"""Chat models built from a key the user brings with their request.

The key lives for one request: it is passed to the SDK client and dropped.
Embeddings stay on the server's embedder, so search works the same whichever
model answers (Anthropic has no embeddings API, and vectors from different
embedders cannot be compared).
"""

from __future__ import annotations

from typing import Literal, get_args

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from doc_agent.llm.anthropic_adapter import DEFAULT_CLAUDE_MODEL, AnthropicChatModel
from doc_agent.llm.openai_adapter import DEFAULT_CHAT_MODEL, OpenAIChatModel
from doc_agent.ports import ChatModel, ModelAuthError

Provider = Literal["openai", "anthropic"]
PROVIDERS: tuple[str, ...] = get_args(Provider)
DEFAULT_MODELS: dict[str, str] = {"openai": DEFAULT_CHAT_MODEL, "anthropic": DEFAULT_CLAUDE_MODEL}


def chat_model_for(provider: Provider, key: str, model: str | None = None) -> ChatModel:
    # SDK retries are off: the agent retries once itself and records each attempt
    if provider == "openai":
        return OpenAIChatModel(
            AsyncOpenAI(api_key=key, max_retries=0, timeout=30), model or DEFAULT_CHAT_MODEL
        )
    return AnthropicChatModel(
        AsyncAnthropic(api_key=key, max_retries=0, timeout=60), model or DEFAULT_CLAUDE_MODEL
    )


async def verify_key(provider: Provider, key: str) -> None:
    """Lists models, which costs nothing, to check a key before it is used."""
    try:
        if provider == "openai":
            await AsyncOpenAI(api_key=key, max_retries=0, timeout=15).models.list()
        else:
            await AsyncAnthropic(api_key=key, max_retries=0, timeout=15).models.list(limit=1)
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
    ) as e:
        raise ModelAuthError(f"{provider} rejected the API key") from e
