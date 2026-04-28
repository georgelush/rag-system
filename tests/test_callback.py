"""
tests/test_callback.py — Tests for webhook callback delivery and job status tracking.

Covers:
  - _send_callback returns True on HTTP 2xx response
  - _send_callback returns False on HTTP 4xx/5xx response
  - _send_callback returns False on network exception
  - _send_callback returns False when callback_url is None
  - HMAC-SHA256 signature in X-Vendor-Signature header when WEBHOOK_SECRET set
  - No signature header when WEBHOOK_SECRET not set
  - update_callback_status persists "sent" on the job
  - update_callback_status persists "failed" on the job
"""

import json
import hmac
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── _send_callback tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_callback_returns_true_on_2xx():
    from rag.ingest import _send_callback

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        result = await _send_callback("https://example.com/webhook", {"job_id": "j-1", "status": "done"})

    assert result is True


@pytest.mark.asyncio
async def test_send_callback_returns_true_on_201():
    from rag.ingest import _send_callback

    mock_resp = MagicMock()
    mock_resp.status_code = 201

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        result = await _send_callback("https://example.com/webhook", {"job_id": "j-1"})

    assert result is True


@pytest.mark.asyncio
async def test_send_callback_returns_false_on_4xx():
    from rag.ingest import _send_callback

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        result = await _send_callback("https://example.com/webhook", {"job_id": "j-1"})

    assert result is False


@pytest.mark.asyncio
async def test_send_callback_returns_false_on_5xx():
    from rag.ingest import _send_callback

    mock_resp = MagicMock()
    mock_resp.status_code = 500

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        result = await _send_callback("https://example.com/webhook", {"job_id": "j-1"})

    assert result is False


@pytest.mark.asyncio
async def test_send_callback_returns_false_on_network_exception():
    from rag.ingest import _send_callback
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        result = await _send_callback("https://example.com/webhook", {"job_id": "j-1"})

    assert result is False


@pytest.mark.asyncio
async def test_send_callback_returns_false_when_url_is_none():
    from rag.ingest import _send_callback
    result = await _send_callback(None, {"job_id": "j-1"})
    assert result is False


# ── HMAC signature ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_callback_includes_hmac_signature_when_secret_set():
    """When WEBHOOK_SECRET is set, X-Vendor-Signature header with sha256=... must be sent."""
    import rag.ingest as ingest_mod

    payload = {"job_id": "j-1", "status": "done"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    expected_sig = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()

    captured_headers: dict = {}

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    async def _capture_post(url, *, content, headers, **kwargs):
        captured_headers.update(headers)
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = _capture_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.object(ingest_mod, "WEBHOOK_SECRET", "test-secret"), \
         patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        await ingest_mod._send_callback("https://example.com/webhook", payload)

    assert "X-Vendor-Signature" in captured_headers
    assert captured_headers["X-Vendor-Signature"] == f"sha256={expected_sig}"


@pytest.mark.asyncio
async def test_send_callback_no_signature_header_when_secret_empty():
    """When WEBHOOK_SECRET is empty/None, no X-Vendor-Signature header should be sent."""
    import rag.ingest as ingest_mod

    captured_headers: dict = {}
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    async def _capture_post(url, *, content, headers, **kwargs):
        captured_headers.update(headers)
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = _capture_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.object(ingest_mod, "WEBHOOK_SECRET", ""), \
         patch("rag.ingest.httpx.AsyncClient", return_value=mock_client):
        await ingest_mod._send_callback("https://example.com/webhook", {"job_id": "j-1"})

    assert "X-Vendor-Signature" not in captured_headers


# ── update_callback_status persistence ───────────────────────────────────────

@pytest.mark.asyncio
async def test_update_callback_status_persists_sent():
    """update_callback_status('sent') → job.callback_status == 'sent' in Redis."""
    from rag.jobs import create_job, get_job, update_callback_status
    from rag.models import IngestJob

    store: dict[str, bytes] = {}

    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=lambda k: store.get(k))
    redis.set = AsyncMock(side_effect=lambda k, v, ex=None: store.__setitem__(k, v if isinstance(v, bytes) else v.encode()))

    job = await create_job(redis, namespace_id="ns-1", source_id="src-1")
    await update_callback_status(redis, job.job_id, "sent")

    loaded = await get_job(redis, job.job_id)
    assert loaded is not None
    assert loaded.callback_status == "sent"


@pytest.mark.asyncio
async def test_update_callback_status_persists_failed():
    from rag.jobs import create_job, get_job, update_callback_status

    store: dict = {}

    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=lambda k: store.get(k))
    redis.set = AsyncMock(side_effect=lambda k, v, ex=None: store.__setitem__(k, v if isinstance(v, bytes) else v.encode()))

    job = await create_job(redis, namespace_id="ns-1", source_id="src-1")
    await update_callback_status(redis, job.job_id, "failed")

    loaded = await get_job(redis, job.job_id)
    assert loaded is not None
    assert loaded.callback_status == "failed"


@pytest.mark.asyncio
async def test_update_callback_status_noop_for_missing_job():
    """update_callback_status on unknown job_id should not raise."""
    from rag.jobs import update_callback_status

    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    # Should not raise
    await update_callback_status(redis, "nonexistent-job", "sent")
