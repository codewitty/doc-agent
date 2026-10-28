import pytest

from doc_agent.config import DEV_TOKENS, ConfigError, Settings


def test_defaults_to_keyless_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "DOC_AGENT_MODEL", "DOC_AGENT_EMBEDDER", "DOC_AGENT_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert (s.model, s.embedder) == ("heuristic", "hashing")
    assert s.tokens == DEV_TOKENS


def test_uses_openai_when_a_key_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DOC_AGENT_MODEL", raising=False)
    monkeypatch.delenv("DOC_AGENT_EMBEDDER", raising=False)
    assert Settings.from_env().model == "openai"


def test_prod_refuses_to_start_with_dev_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_AGENT_ENV", "prod")
    monkeypatch.delenv("DOC_AGENT_TOKENS", raising=False)
    with pytest.raises(ConfigError, match="DOC_AGENT_TOKENS"):
        Settings.from_env()


def test_parses_token_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_AGENT_TOKENS", "t1:alice, t2:bob")
    assert Settings.from_env().tokens == {"t1": "alice", "t2": "bob"}


@pytest.mark.parametrize("raw", ["justtoken", ":alice", "t1:"])
def test_rejects_malformed_tokens(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("DOC_AGENT_TOKENS", raw)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_openai_backend_requires_a_key() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings(model="openai").validate()
