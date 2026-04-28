"""rag_server.py — FastAPI entry point for the RAG Example service."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
load_dotenv()  # Must be before any rag.* imports so config.py reads env vars correctly

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from rag.logging_config import setup_logging
from rag.rate_limiter import RateLimiterMiddleware
from rag.metrics import MetricsMiddleware, metrics_response
from rag import telemetry
from rag.config import validate_startup

from rag.routers.query_router import router as query_router
from rag.routers.ingest_router import router as ingest_router
from rag.routers.namespaces_router import router as namespaces_router
from rag.routers.health_router import router as health_router
from rag.routers.stream_router import router as stream_router
from rag.routers.tokens_router import router as tokens_router

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

import logging
log = logging.getLogger("rag.server")

QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
REDIS_URL:  str = os.environ.get("REDIS_URL",  "redis://localhost:6379")
PORT:       int = int(os.environ.get("RAG_PORT", "8080"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Env var validation ─────────────────────────────────────────────────────
    for warning in validate_startup():
        log.warning("Config warning: %s", warning)

    log.info("Service starting", extra={"qdrant": QDRANT_URL, "redis": REDIS_URL})
    app.state.start_time = time.time()
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL)
    app.state.redis  = await aioredis.from_url(REDIS_URL, decode_responses=False)

    # ── Startup health probes (warn-only — service starts regardless) ──────────
    try:
        await app.state.qdrant.get_collections()
        log.info("Qdrant connection verified")
    except Exception as exc:
        log.warning("Qdrant unreachable at startup", extra={"error": str(exc)})
    try:
        await app.state.redis.ping()
        log.info("Redis connection verified")
    except Exception as exc:
        log.warning("Redis unreachable at startup", extra={"error": str(exc)})

    # ── Start embedded ingest workers ──────────────────────────────────────────
    from rag.worker import run_workers
    _worker_task = asyncio.create_task(
        run_workers(app.state.qdrant, app.state.redis)
    )

    yield

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    _worker_task.cancel()
    await asyncio.gather(_worker_task, return_exceptions=True)
    log.info("Service shutting down")
    telemetry.flush()
    await app.state.qdrant.close()
    await app.state.redis.aclose()


app = FastAPI(
    title="RAG Example Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimiterMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_route("/metrics", metrics_response)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "internal_error", "message": str(exc.detail), "request_id": ""}},
    )


app.include_router(query_router)
app.include_router(stream_router)
app.include_router(ingest_router)
app.include_router(namespaces_router)
app.include_router(health_router)
app.include_router(tokens_router)


@app.get("/v1/openapi.json", include_in_schema=False)
async def openapi_v1() -> JSONResponse:
    """Spec §15: openapi.yaml exposed live at GET /v1/openapi.json."""
    return JSONResponse(
        get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )
    )


if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="0.0.0.0", port=PORT, reload=False)
