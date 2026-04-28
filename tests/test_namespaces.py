"""
tests/test_namespaces.py — Namespace and source management endpoint tests.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import AUTH_HEADERS


# ── Helper ─────────────────────────────────────────────────────────────────────

def _idempotency_headers(key: str = "idem-001") -> dict:
    return {**AUTH_HEADERS, "Idempotency-Key": key}


# ── GET /v1/namespaces/{id}/stats ─────────────────────────────────────────────

async def test_get_stats_not_found_returns_404(client, mock_qdrant):
    """Namespace that doesn't exist → 404."""
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    resp = await client.get(
        "/v1/namespaces/nonexistent/stats",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_get_stats_existing_namespace(client, mock_qdrant):
    """Existing namespace (collection present in Qdrant) → 200 with stats."""
    col_mock = MagicMock()
    col_mock.name = "tenant-test__my-namespace"
    mock_qdrant.get_collections.return_value = MagicMock(collections=[col_mock])

    count_mock = MagicMock()
    count_mock.count = 42
    mock_qdrant.count.return_value = count_mock

    scroll_mock = ([], None)  # empty scroll
    mock_qdrant.scroll.return_value = scroll_mock

    resp = await client.get(
        "/v1/namespaces/my-namespace/stats",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace_id"] == "my-namespace"
    assert "chunk_count" in data


# ── DELETE /v1/namespaces/{id}/sources/{source_id} ────────────────────────────

async def test_delete_source_not_found_returns_404(client, mock_qdrant):
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    resp = await client.delete(
        "/v1/namespaces/ns/sources/source-1",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404


async def test_delete_source_existing_returns_204(client, mock_qdrant):
    col_mock = MagicMock()
    col_mock.name = "tenant-test__ns"
    mock_qdrant.get_collections.return_value = MagicMock(collections=[col_mock])

    resp = await client.delete(
        "/v1/namespaces/ns/sources/source-1",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 204


# ── DELETE /v1/namespaces/{id} ────────────────────────────────────────────────

async def test_delete_namespace_not_found_returns_404(client, mock_qdrant):
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    resp = await client.delete(
        "/v1/namespaces/nonexistent",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404


async def test_delete_namespace_existing_returns_202(client, mock_qdrant):
    col_mock = MagicMock()
    col_mock.name = "tenant-test__my-ns"
    mock_qdrant.get_collections.return_value = MagicMock(collections=[col_mock])

    resp = await client.delete(
        "/v1/namespaces/my-ns",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


# ── POST /v1/ingest ────────────────────────────────────────────────────────────

async def test_ingest_missing_idempotency_key_returns_400(client):
    body = {
        "namespace_id": "ns",
        "source_id": "src",
        "source_type": "file",
    }
    resp = await client.post("/v1/ingest", json=body, headers=AUTH_HEADERS)
    assert resp.status_code == 400
    assert "Idempotency-Key" in resp.json()["error"]["message"]


async def test_ingest_file_mode_returns_202(client, mock_redis):
    """Ingest endpoint accepts request and returns 202 immediately."""
    from unittest.mock import patch, AsyncMock

    body = {
        "namespace_id": "ns",
        "source_id": "src-new",
        "source_type": "file",
    }

    # Mock job creation
    from rag.models import IngestJob, IngestStatus
    from datetime import datetime, timezone

    mock_job = IngestJob(
        job_id="job-001",
        namespace_id="ns",
        source_id="src-new",
        status=IngestStatus.queued,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )

    with (
        patch("rag.routers.ingest_router.create_job", AsyncMock(return_value=mock_job)),
        patch("rag.routers.ingest_router.enqueue_ingest", AsyncMock()),
    ):
        resp = await client.post(
            "/v1/ingest",
            json=body,
            headers={**AUTH_HEADERS, "Idempotency-Key": "idem-test-001"},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "job-001"
    assert data["status"] == "queued"


# ── GET /v1/ingest/{job_id} ────────────────────────────────────────────────────

async def test_get_ingest_status_not_found(client, mock_redis):
    mock_redis.get.return_value = None

    resp = await client.get(
        "/v1/ingest/unknown-job-id",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404


# ── Idempotency deduplication ─────────────────────────────────────────────────

async def test_ingest_same_idempotency_key_returns_same_job_id(client):
    """Two POST /v1/ingest with identical Idempotency-Key must return the same job_id."""
    from unittest.mock import AsyncMock, patch
    from rag.models import IngestJob, IngestStatus, IngestProgress

    body = {
        "source_id": "src-idem-dedup",
        "namespace_id": "ns-test",
        "source_type": "url",
        "url": "https://example.com/doc.pdf",
    }

    first_job = IngestJob(
        job_id="job-dedup-001",
        namespace_id="ns-test",
        source_id="src-idem-dedup",
        status=IngestStatus.queued,
        progress=IngestProgress(stage=IngestStatus.queued, percent=0, chunks_created=0),
        submitted_at="2025-01-01T00:00:00Z",
    )

    async def _idempotent_create_job(*args, idempotency_key=None, **kwargs):
        return first_job  # Always return the SAME job regardless of call count

    with patch("rag.routers.ingest_router.create_job", side_effect=_idempotent_create_job), \
         patch("rag.routers.ingest_router.enqueue_ingest", AsyncMock()):

        resp1 = await client.post(
            "/v1/ingest",
            json=body,
            headers={**AUTH_HEADERS, "Idempotency-Key": "unique-key-xyz"},
        )
        resp2 = await client.post(
            "/v1/ingest",
            json=body,
            headers={**AUTH_HEADERS, "Idempotency-Key": "unique-key-xyz"},
        )

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    assert resp1.json()["job_id"] == resp2.json()["job_id"] == "job-dedup-001"


async def test_ingest_different_idempotency_keys_return_different_jobs(client):
    """Different Idempotency-Keys must produce separate jobs."""
    from unittest.mock import AsyncMock, patch
    from rag.models import IngestJob, IngestStatus, IngestProgress

    body = {
        "source_id": "src-multi",
        "namespace_id": "ns-test",
        "source_type": "url",
        "url": "https://example.com/doc.pdf",
    }

    jobs = [
        IngestJob(
            job_id=f"job-multi-{i}",
            namespace_id="ns-test",
            source_id="src-multi",
            status=IngestStatus.queued,
            progress=IngestProgress(stage=IngestStatus.queued, percent=0, chunks_created=0),
            submitted_at="2025-01-01T00:00:00Z",
        )
        for i in range(2)
    ]
    call_idx = [0]

    async def _new_job(*args, idempotency_key=None, **kwargs):
        j = jobs[call_idx[0]]
        call_idx[0] += 1
        return j

    with patch("rag.routers.ingest_router.create_job", side_effect=_new_job), \
         patch("rag.routers.ingest_router.enqueue_ingest", AsyncMock()):

        resp1 = await client.post(
            "/v1/ingest",
            json=body,
            headers={**AUTH_HEADERS, "Idempotency-Key": "key-A"},
        )
        resp2 = await client.post(
            "/v1/ingest",
            json=body,
            headers={**AUTH_HEADERS, "Idempotency-Key": "key-B"},
        )

    assert resp1.json()["job_id"] != resp2.json()["job_id"]
