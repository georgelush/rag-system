"""src/rag/routers/ingest_router.py — POST /v1/ingest, GET /v1/ingest/{job_id}"""

import json
from typing import Annotated
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from pydantic import ValidationError

from rag.auth import require_auth, _make_error
from rag.jobs import create_job, get_job
from rag.models import ErrorCode, IngestJob, IngestRequest, IngestStatus
from rag.queue import enqueue_ingest
from rag.rbac import AuthResult

router = APIRouter()

_TERMINAL = {IngestStatus.done, IngestStatus.failed, IngestStatus.cancelled}


@router.post("/v1/ingest", response_model=IngestJob, status_code=202)
async def ingest(
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> IngestJob:
    r: Redis = request.app.state.redis

    content_type = request.headers.get("content-type", "")

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            payload_str = form.get("payload", "{}")
            file_field = form.get("file")
            file_bytes: bytes | None = await file_field.read() if file_field else None
            req = IngestRequest.model_validate(json.loads(payload_str))
        else:
            body = await request.json()
            req = IngestRequest.model_validate(body)
            file_bytes = None
    except ValidationError:
        raise _make_error(ErrorCode.validation_error, "Request body validation failed", auth.request_id, 422)

    if not idempotency_key:
        raise _make_error(ErrorCode.invalid_request, "Missing required header: Idempotency-Key", auth.request_id, 400)

    auth.assert_can_write(req.namespace_id)

    job = await create_job(r, req.namespace_id, req.source_id, idempotency_key)
    await enqueue_ingest(
        r,
        job_id=job.job_id,
        tenant_id=auth.tenant_id,
        namespace_id=req.namespace_id,
        source_id=req.source_id,
        req_dict=req.model_dump(mode="json"),
        file_bytes_hex=file_bytes.hex() if file_bytes else None,
    )
    return job


@router.get("/v1/ingest/{job_id}", response_model=IngestJob)
async def get_ingest_status(
    job_id: str,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> JSONResponse:
    r: Redis = request.app.state.redis
    job = await get_job(r, job_id)

    if job is None:
        raise _make_error(ErrorCode.not_found, f"Job {job_id} not found", auth.request_id, 404)

    headers: dict[str, str] = {}
    if job.status not in _TERMINAL:
        headers["Retry-After"] = "5"

    return JSONResponse(
        content=json.loads(job.model_dump_json()),
        headers=headers,
    )