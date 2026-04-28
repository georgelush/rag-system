"""
src/rag/jobs.py — Redis-backed job store for async ingestion tracking.

Spec §4.2: POST /v1/ingest returns 202 with job_id immediately.
Spec §4.3: GET /v1/ingest/{job_id} polls current status.
Spec §2:   Idempotency-Key → same key = same job_id (no duplicate processing).

Each job is stored in Redis as a JSON hash under key: rag:job:{job_id}
Idempotency map stored under key: rag:idem:{idempotency_key} → job_id
TTL: 7 days (jobs auto-expire from Redis).
"""

import json
import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis

from rag.models import IngestJob, IngestProgress, IngestStatus

# ── Constants ──────────────────────────────────────────────────────────────────

JOB_PREFIX  = "rag:job:"    # rag:job:{job_id}    → JSON job data
IDEM_PREFIX = "rag:idem:"   # rag:idem:{idem_key} → job_id
JOB_TTL_SEC = 60 * 60 * 24 * 7  # 7 days


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"


def _idem_key(idempotency_key: str) -> str:
    return f"{IDEM_PREFIX}{idempotency_key}"


# ── CRUD ───────────────────────────────────────────────────────────────────────

async def create_job(
    r: aioredis.Redis,
    namespace_id: str,
    source_id: str,
    idempotency_key: str | None = None,
) -> IngestJob:
    """
    Creates a new job in Redis and returns it.
    If idempotency_key already exists → returns the existing job (no duplicate).
    """
    if idempotency_key:
        existing_id = await r.get(_idem_key(idempotency_key))
        if existing_id:
            return await get_job(r, existing_id.decode())

    job_id = f"j_{uuid.uuid4().hex[:12]}"
    job = IngestJob(
        job_id=job_id,
        namespace_id=namespace_id,
        source_id=source_id,
        status=IngestStatus.queued,
        progress=IngestProgress(
            stage=IngestStatus.queued,
            percent=0,
            chunks_created=0,
        ),
        submitted_at=_now(),
        estimated_completion_at=None,
        completed_at=None,
        error=None,
    )

    await r.set(_job_key(job_id), job.model_dump_json(), ex=JOB_TTL_SEC)

    if idempotency_key:
        await r.set(_idem_key(idempotency_key), job_id, ex=JOB_TTL_SEC)

    return job


async def get_job(r: aioredis.Redis, job_id: str) -> IngestJob | None:
    """Returns job or None if not found."""
    raw = await r.get(_job_key(job_id))
    if not raw:
        return None
    return IngestJob.model_validate_json(raw)


async def update_job(
    r: aioredis.Redis,
    job_id: str,
    status: IngestStatus,
    percent: int = 0,
    chunks_created: int = 0,
    error: dict | None = None,
) -> None:
    """Updates job status + progress in Redis."""
    job = await get_job(r, job_id)
    if not job:
        return

    job.status = status
    job.progress = IngestProgress(
        stage=status,
        percent=percent,
        chunks_created=chunks_created,
    )

    if status in (IngestStatus.done, IngestStatus.failed, IngestStatus.cancelled):
        job.completed_at = _now()

    if error:
        from rag.models import IngestError
        job.error = error if isinstance(error, IngestError) else IngestError(**error)

    await r.set(_job_key(job_id), job.model_dump_json(), ex=JOB_TTL_SEC)


async def update_callback_status(
    r: aioredis.Redis,
    job_id: str,
    status: str,
) -> None:
    """Record webhook callback delivery status on the job.

    status: 'sent' (HTTP 2xx received), 'failed' (all retries exhausted).
    """
    job = await get_job(r, job_id)
    if not job:
        return
    job.callback_status = status
    await r.set(_job_key(job_id), job.model_dump_json(), ex=JOB_TTL_SEC)