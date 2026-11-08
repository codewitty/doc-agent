from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from doc_agent.agent.orchestrator import InvalidRunState
from doc_agent.agent.tools import is_stale
from doc_agent.api.schemas import DecisionRequest, DocumentView, RunView, StartRunRequest
from doc_agent.config import Settings
from doc_agent.container import Container, build_container
from doc_agent.data.ingest import IngestionError
from doc_agent.fixtures import seed_fixtures
from doc_agent.logging_config import configure_logging

log = logging.getLogger("doc_agent.api")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _dependency_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [sqlite3.OperationalError, openai.APIError, TimeoutError]
    try:
        import psycopg
        import psycopg_pool

        errors += [psycopg.OperationalError, psycopg_pool.PoolTimeout]
    except ImportError:
        pass
    return tuple(errors)


DEPENDENCY_ERRORS = _dependency_errors()

_bearer = HTTPBearer(auto_error=False)


# Dependencies live at module level: with postponed annotations FastAPI
# resolves these aliases from module globals, not from create_app's scope.
def get_container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


Deps = Annotated[Container, Depends(get_container)]


def current_user(
    c: Deps,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    user = c.settings.tokens.get(creds.credentials) if creds else None
    if user is None:
        raise HTTPException(401, "Missing or invalid bearer token")
    return user


User = Annotated[str, Depends(current_user)]


def create_app(container: Container | None = None, seed: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is None:
            configure_logging()
        c = container or build_container(Settings.from_env())
        app.state.container = c
        if seed:
            for user in c.settings.seed_users:
                added = await run_in_threadpool(seed_fixtures, c.ingestion, user)
                log.info("fixtures.seeded", extra={"user_id": user, "added": added})
        log.info(
            "app.started",
            extra={"model": c.model.name, "embedder": c.embedder.name, "env": c.settings.env},
        )
        yield

    app = FastAPI(title="doc-agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        if request.url.path.startswith("/api/"):
            log.info(
                "http.request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return response

    @app.exception_handler(IngestionError)
    async def ingestion_error(request: Request, e: IngestionError) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(InvalidRunState)
    async def invalid_state(request: Request, e: InvalidRunState) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, e: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        if isinstance(e, DEPENDENCY_ERRORS):
            log.error("dependency.unavailable", exc_info=e, extra={"request_id": request_id})
            return JSONResponse(
                {
                    "detail": "A backing service is unavailable. Try again shortly.",
                    "request_id": request_id,
                },
                status_code=503,
                headers={"x-request-id": request_id or ""},
            )
        log.exception("http.unhandled", extra={"request_id": request_id})
        return JSONResponse(
            {"detail": "Internal server error", "request_id": request_id},
            status_code=500,
            headers={"x-request-id": request_id or ""},
        )

    @app.get("/api/health")
    def health(c: Deps) -> dict[str, str]:
        return {
            "status": "ok",
            "model": c.model.name,
            "embedder": c.embedder.name,
            "database": c.settings.database_url.split(":", 1)[0],
        }

    @app.get("/api/me")
    def me(user: User) -> dict[str, str]:
        return {"user_id": user}

    @app.get("/api/documents")
    def list_documents(user: User, c: Deps) -> list[DocumentView]:
        now = datetime.now(UTC)
        return [
            DocumentView.of(d, is_stale(d.created_at, now))
            for d in c.documents.list_documents(user)
        ]

    @app.post("/api/documents", status_code=201)
    async def upload_document(
        user: User, c: Deps, file: Annotated[UploadFile, File()]
    ) -> DocumentView:
        data = await file.read()
        doc = await run_in_threadpool(c.ingestion.ingest, user, file.filename or "upload", data)
        return DocumentView.of(doc, stale=False)

    @app.post("/api/runs", status_code=201)
    async def start_run(body: StartRunRequest, user: User, c: Deps) -> RunView:
        return RunView.of(await c.agent().start(user, body.question))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, user: User, c: Deps) -> RunView:
        run = c.agent().get(user, run_id)
        # Another user's run is reported as missing, not forbidden, so ids don't leak
        if run is None:
            raise HTTPException(404, "Run not found")
        return RunView.of(run)

    @app.post("/api/runs/{run_id}/decision")
    async def decide(run_id: str, body: DecisionRequest, user: User, c: Deps) -> RunView:
        run = await c.agent().decide(user, run_id, body.approve, body.note)
        if run is None:
            raise HTTPException(404, "Run not found")
        return RunView.of(run)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
