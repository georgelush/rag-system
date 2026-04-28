"""src/rag/routers/namespaces_router.py"""

import uuid
from typing import Annotated
from fastapi import APIRouter, Depends, Request
from qdrant_client import AsyncQdrantClient

from rag.auth import require_auth, _make_error
from rag.models import DeleteJobResponse, ErrorCode, NamespaceStats
from rag.vector_store import collection_name, delete_source, delete_namespace, get_stats_cached
from rag.rbac import AuthResult

router = APIRouter()


async def _collection_exists(qdrant: AsyncQdrantClient, col: str) -> bool:
    existing = {c.name for c in (await qdrant.get_collections()).collections}
    return col in existing


@router.delete("/v1/namespaces/{namespace_id}/sources/{source_id}", status_code=204)
async def delete_source_endpoint(
    namespace_id: str,
    source_id: str,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> None:
    auth.assert_can_write(namespace_id)
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    col = collection_name(auth.tenant_id, namespace_id)
    if not await _collection_exists(qdrant, col):
        raise _make_error(ErrorCode.not_found, f"Namespace {namespace_id} not found", auth.request_id, 404)
    await delete_source(qdrant, col, source_id)


@router.delete("/v1/namespaces/{namespace_id}", status_code=202)
async def delete_namespace_endpoint(
    namespace_id: str,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> DeleteJobResponse:
    auth.assert_can_admin(namespace_id)
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    col = collection_name(auth.tenant_id, namespace_id)
    if not await _collection_exists(qdrant, col):
        raise _make_error(ErrorCode.not_found, f"Namespace {namespace_id} not found", auth.request_id, 404)
    await delete_namespace(qdrant, col)
    return DeleteJobResponse(
        job_id=f"j_{uuid.uuid4().hex[:12]}",
        status="queued",
        sla="24h",
    )


@router.get("/v1/namespaces/{namespace_id}/stats", response_model=NamespaceStats)
async def get_namespace_stats(
    namespace_id: str,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> NamespaceStats:
    auth.assert_can_read(namespace_id)
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    col = collection_name(auth.tenant_id, namespace_id)
    stats = await get_stats_cached(qdrant, col, namespace_id, redis=request.app.state.redis)
    if stats is None:
        raise _make_error(ErrorCode.not_found, f"Namespace {namespace_id} not found", auth.request_id, 404)
    return stats
