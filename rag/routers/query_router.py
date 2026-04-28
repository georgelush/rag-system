"""src/rag/routers/query_router.py — POST /v1/query and POST /v1/query/batch"""

import asyncio
from typing import Annotated
from fastapi import APIRouter, Depends, Request
from qdrant_client import AsyncQdrantClient

from rag.auth import require_auth
from rag.models import QueryRequest, QueryResponse, BatchQueryRequest, BatchQueryResponse
from rag.query import run_query
from rag.rbac import AuthResult
from rag.config import BATCH_QUERY_MAX

router = APIRouter()


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> QueryResponse:
    for ns in req.namespaces:
        auth.assert_can_read(ns)
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    redis = request.app.state.redis
    return await run_query(qdrant, req, auth.request_id, auth.tenant_id, redis=redis)


@router.post("/v1/query/batch", response_model=BatchQueryResponse)
async def query_batch(
    req: BatchQueryRequest,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> BatchQueryResponse:
    """Run up to BATCH_QUERY_MAX queries in parallel.

    Results are ordered to match the input ``queries`` list (same index).
    Per-query errors are returned inline as ``{"error": {"code": ..., "message": ...}}``
    so a single failed query does not abort the whole batch.
    """
    if len(req.queries) > BATCH_QUERY_MAX:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "validation_error",
                    "message": f"batch size {len(req.queries)} exceeds BATCH_QUERY_MAX={BATCH_QUERY_MAX}",
                    "request_id": auth.request_id,
                }
            },
        )

    # Verify namespace read permissions for all queries up front
    for q in req.queries:
        for ns in q.namespaces:
            auth.assert_can_read(ns)

    qdrant: AsyncQdrantClient = request.app.state.qdrant
    redis = request.app.state.redis

    async def _run_one(q: QueryRequest):
        try:
            return await run_query(qdrant, q, auth.request_id, auth.tenant_id, redis=redis)
        except Exception as exc:
            return {"error": {"code": "internal_error", "message": str(exc), "request_id": auth.request_id}}

    results = await asyncio.gather(*[_run_one(q) for q in req.queries])
    return BatchQueryResponse(results=list(results))
