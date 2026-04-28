"""src/rag/routers/health_router.py"""

import os
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient

from rag.models import HealthState, HealthStatus

router = APIRouter()

_VERSION = os.environ.get("RAG_VERSION", "1.0.0")


@router.get("/v1/health", response_model=HealthStatus)
async def health(request: Request) -> JSONResponse:
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    r: Redis = request.app.state.redis

    qdrant_state = HealthState.ok
    redis_state = HealthState.ok

    try:
        await qdrant.get_collections()
    except Exception:
        qdrant_state = HealthState.down

    try:
        await r.ping()
    except Exception:
        redis_state = HealthState.down

    overall = (
        HealthState.ok
        if qdrant_state == HealthState.ok and redis_state == HealthState.ok
        else HealthState.down
    )

    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = int(time.time() - start_time)

    body = HealthStatus(
        status=overall,
        version=_VERSION,
        uptime_seconds=uptime,
        dependencies={"qdrant": qdrant_state.value, "redis": redis_state.value},
    )

    status_code = 200 if overall == HealthState.ok else 503
    return JSONResponse(
        content=body.model_dump(mode="json"),
        status_code=status_code,
    )
