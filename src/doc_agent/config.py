from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Env = Literal["dev", "test", "prod"]

DEV_TOKENS = {"alice-dev-token": "alice", "bob-dev-token": "bob"}


class ConfigError(RuntimeError):
    pass


def _tokens(raw: str | None) -> dict[str, str]:
    """Parse "token:user,token:user" into {token: user}."""
    if not raw:
        return {}
    pairs = [p.strip().split(":", 1) for p in raw.split(",") if p.strip()]
    if any(len(p) != 2 or not p[0] or not p[1] for p in pairs):
        raise ConfigError("DOC_AGENT_TOKENS must look like token:user,token:user")
    return {t: u for t, u in pairs}


@dataclass(frozen=True)
class Settings:
    env: Env = "dev"
    database_url: str = "sqlite:///doc-agent.db"
    model: Literal["heuristic", "openai"] = "heuristic"
    embedder: Literal["hashing", "openai"] = "hashing"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    tokens: dict[str, str] = field(default_factory=lambda: dict(DEV_TOKENS))
    seed_users: tuple[str, ...] = ("alice",)

    @classmethod
    def from_env(cls) -> Settings:
        env: Env = os.environ.get("DOC_AGENT_ENV", "dev")  # type: ignore[assignment]
        key = os.environ.get("OPENAI_API_KEY") or None
        tokens = _tokens(os.environ.get("DOC_AGENT_TOKENS"))
        if env == "prod" and not tokens:
            raise ConfigError("DOC_AGENT_TOKENS is required when DOC_AGENT_ENV=prod")
        settings = cls(
            env=env,
            database_url=os.environ.get("DATABASE_URL", cls.database_url),
            model=os.environ.get("DOC_AGENT_MODEL", "openai" if key else "heuristic"),  # type: ignore[arg-type]
            embedder=os.environ.get("DOC_AGENT_EMBEDDER", "openai" if key else "hashing"),  # type: ignore[arg-type]
            openai_api_key=key,
            openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL", cls.openai_chat_model),
            tokens=tokens or dict(DEV_TOKENS),
            seed_users=tuple(
                u.strip()
                for u in os.environ.get("DOC_AGENT_SEED_USERS", "alice").split(",")
                if u.strip()
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.model not in ("heuristic", "openai"):
            raise ConfigError(f"Unknown DOC_AGENT_MODEL {self.model!r}")
        if self.embedder not in ("hashing", "openai"):
            raise ConfigError(f"Unknown DOC_AGENT_EMBEDDER {self.embedder!r}")
        if "openai" in (self.model, self.embedder) and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required for the openai model or embedder")
