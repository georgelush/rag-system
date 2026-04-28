"""rag/routers/stream_router.py — POST /v1/query/stream (Server-Sent Events)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient

from rag.auth import require_auth
from rag.models import QueryRequest
from rag.query import run_query_stream
from rag.rbac import AuthResult

router = APIRouter()


@router.post("/v1/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> StreamingResponse:
    """Stream the query response as Server-Sent Events.

    The response body is ``text/event-stream``. Events in order:

    * ``citations`` — JSON array of Citation objects, sent right after retrieval
    * ``token``     — one event per LLM token, ``{"token": "..."}``
    * ``done``      — final metadata: ``{"request_id", "retrieval_strategy", "usage", "latency_ms"}``
    * ``error``     — only on failure: ``{"code", "message"}``

    The caller should close the connection after receiving ``done`` or ``error``.
    """
    for ns in req.namespaces:
        auth.assert_can_read(ns)
    qdrant: AsyncQdrantClient = request.app.state.qdrant

    return StreamingResponse(
        run_query_stream(qdrant, req, auth.request_id, auth.tenant_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":        "keep-alive",
        },
    )
