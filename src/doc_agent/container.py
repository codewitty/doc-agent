"""Builds the object graph from Settings. The only place that picks adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from doc_agent.agent.orchestrator import Agent, Limits
from doc_agent.config import Settings
from doc_agent.data.ingest import IngestionService
from doc_agent.data.sqlite import SQLiteDatabase, SQLiteDocumentRepository, SQLiteRunRepository
from doc_agent.llm import providers
from doc_agent.llm.fakes import HashingEmbedder, HeuristicChatModel
from doc_agent.llm.providers import Provider
from doc_agent.ports import ChatModel, DocumentRepository, Embedder, RunRepository


@dataclass
class Container:
    settings: Settings
    documents: DocumentRepository
    runs: RunRepository
    model: ChatModel
    embedder: Embedder
    limits: Limits = field(default_factory=Limits)
    # Builds a model from a user's own key; tests swap in fakes
    chat_model_for: Callable[[Provider, str], ChatModel] = providers.chat_model_for
    verify_key: Callable[[Provider, str], Awaitable[None]] = providers.verify_key

    @property
    def ingestion(self) -> IngestionService:
        return IngestionService(self.documents, self.embedder)

    def agent(self, model: ChatModel | None = None) -> Agent:
        return Agent(
            model=model or self.model,
            documents=self.documents,
            runs=self.runs,
            embedder=self.embedder,
            limits=self.limits,
        )


def _repositories(url: str) -> tuple[DocumentRepository, RunRepository]:
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        sqlite = SQLiteDatabase(path)
        return SQLiteDocumentRepository(sqlite), SQLiteRunRepository(sqlite)
    if url.startswith(("postgres://", "postgresql://")):
        # Imported lazily so the SQLite path works without the postgres extra
        from doc_agent.data.postgres import (
            PostgresDatabase,
            PostgresDocumentRepository,
            PostgresRunRepository,
        )

        pg = PostgresDatabase(url)
        return PostgresDocumentRepository(pg), PostgresRunRepository(pg)
    raise ValueError(f"Unsupported DATABASE_URL scheme: {url.split(':', 1)[0]}")


def build_container(settings: Settings) -> Container:
    documents, runs = _repositories(settings.database_url)

    model: ChatModel
    embedder: Embedder
    if settings.model == "openai" or settings.embedder == "openai":
        from openai import AsyncOpenAI, OpenAI

        from doc_agent.llm.openai_adapter import OpenAIChatModel, OpenAIEmbedder

    if settings.model == "openai":
        model = OpenAIChatModel(
            AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0), settings.openai_chat_model
        )
    elif settings.model == "anthropic":
        assert settings.anthropic_api_key is not None
        model = providers.chat_model_for(
            "anthropic", settings.anthropic_api_key, settings.anthropic_chat_model
        )
    else:
        model = HeuristicChatModel()

    if settings.embedder == "openai":
        embedder = OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key, timeout=15))
    else:
        embedder = HashingEmbedder()

    return Container(
        settings=settings, documents=documents, runs=runs, model=model, embedder=embedder
    )
