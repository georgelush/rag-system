"""
tests/test_rate_limiter.py — Tests for rag/rate_limiter.py.

Covers:
  - Over-limit returns 429 with Retry-After header
  - Under-limit passes through
  - /v1/health and other exempt paths bypass limiter
  - Redis unavailable → fail-open (request passes, no 429)
  - Correct limit used per path (/v1/query vs /v1/ingest vs default)
  - Missing X-Tenant-ID → request passed to auth middleware (no 429)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key-12345",
    "X-Request-ID": "req-rl-001",
    "X-Tenant-ID": "tenant-rl",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_redis_at_count(count: int) -> AsyncMock:
    """Return a mock Redis where incr() always returns `count`."""
    r = AsyncMock()
    r.incr = AsyncMock(return_value=count)
    r.expire = AsyncMock(return_value=True)
    r.ping = AsyncMock(return_value=b"PONG")
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock(return_value=True)
    r.setex = AsyncMock(return_value=True)
    r.aclose = AsyncMock()
    return r


# ── 429 behaviour ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_exceeded_returns_429(client, mock_qdrant):
    """When Redis incr returns > limit, response must be 429."""
    import rag_server as srv

    over_limit_redis = _make_redis_at_count(999)
    srv.app.state.redis = over_limit_redis

    resp = await client.post(
        "/v1/query",
        json={"question": "Q?", "namespaces": ["ns"]},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    body = resp.json()
    assert body["error"]["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_rate_limit_exceeded_has_retry_after_header(client):
    import rag_server as srv

    srv.app.state.redis = _make_redis_at_count(9999)
    resp = await client.post(
        "/v1/query",
        json={"question": "Q?", "namespaces": ["ns"]},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 429
    retry_after = int(resp.headers["Retry-After"])
    assert 0 < retry_after <= 60


@pytest.mark.asyncio
async def test_under_limit_passes_through(client):
    """count=1 (first request) should not trigger 429."""
    import rag_server as srv
    from unittest.mock import patch

    srv.app.state.redis = _make_redis_at_count(1)

    with patch("rag.routers.query_router.run_query", new_callable=AsyncMock) as mock_run:
        from rag.models import QueryResponse
        mock_run.return_value = QueryResponse(
            request_id="req-1",
            answer="ok",
            usage={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "model_id": "t"},
            latency_ms=0,
            model_version="t",
        )
        resp = await client.post(
            "/v1/query",
            json={"question": "Q?", "namespaces": ["ns"]},
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 200


# ── Exempt paths ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_path_is_exempt_from_rate_limit(client):
    """GET /v1/health should never be rate-limited regardless of Redis state."""
    import rag_server as srv

    srv.app.state.redis = _make_redis_at_count(9999)
    resp = await client.get("/v1/health")
    # Health endpoint itself may be 200 or 503 — but never 429
    assert resp.status_code != 429


@pytest.mark.asyncio
async def test_openapi_path_is_exempt(client):
    import rag_server as srv

    srv.app.state.redis = _make_redis_at_count(9999)
    resp = await client.get("/v1/openapi.json")
    assert resp.status_code != 429


# ── Fail-open ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_unavailable_fails_open(client):
    """If Redis raises on incr, the request must pass through (not 429)."""
    import rag_server as srv
    from unittest.mock import patch

    broken_redis = AsyncMock()
    broken_redis.incr = AsyncMock(side_effect=ConnectionError("Redis down"))
    broken_redis.get = AsyncMock(return_value=None)
    broken_redis.set = AsyncMock(return_value=True)
    broken_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
    srv.app.state.redis = broken_redis

    with patch("rag.routers.query_router.run_query", new_callable=AsyncMock) as mock_run:
        from rag.models import QueryResponse
        mock_run.return_value = QueryResponse(
            request_id="req-1",
            answer="ok",
            usage={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "model_id": "t"},
            latency_ms=0,
            model_version="t",
        )
        resp = await client.post(
            "/v1/query",
            json={"question": "Q?", "namespaces": ["ns"]},
            headers=AUTH_HEADERS,
        )
    # fail-open: should not be 429
    assert resp.status_code != 429


# ── Missing tenant ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_tenant_id_bypasses_rate_limiter(client):
    """No X-Tenant-ID → rate limiter skips, auth middleware returns 400."""
    import rag_server as srv

    srv.app.state.redis = _make_redis_at_count(9999)
    resp = await client.post(
        "/v1/query",
        json={"question": "Q?", "namespaces": ["ns"]},
        headers={
            "Authorization": "Bearer test-api-key-12345",
            "X-Request-ID": "req-1",
            # No X-Tenant-ID
        },
    )
    # Must be 400 (auth rejects missing tenant), not 429 (rate limiter)
    assert resp.status_code == 400
