"""
tests/test_metrics.py — Tests for the Prometheus /metrics endpoint.

Covers:
  - GET /metrics returns HTTP 200
  - Content-Type contains text/plain
  - Response body contains expected metric names
  - Core rag_ counters are present in the output
"""

import pytest


# ── Prometheus /metrics endpoint ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_metrics_content_type_is_text_plain(client):
    resp = await client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_metrics_contains_http_requests_counter(client):
    resp = await client.get("/metrics")
    assert "rag_http_requests_total" in resp.text


@pytest.mark.asyncio
async def test_metrics_contains_ingest_retries_counter(client):
    resp = await client.get("/metrics")
    assert "rag_ingest_retries_total" in resp.text


@pytest.mark.asyncio
async def test_metrics_contains_namespace_queries_counter(client):
    resp = await client.get("/metrics")
    assert "rag_namespace_queries_total" in resp.text


@pytest.mark.asyncio
async def test_metrics_contains_cache_hits_counter(client):
    resp = await client.get("/metrics")
    assert "rag_cache_hits_total" in resp.text


@pytest.mark.asyncio
async def test_metrics_contains_cache_misses_counter(client):
    resp = await client.get("/metrics")
    assert "rag_cache_misses_total" in resp.text


@pytest.mark.asyncio
async def test_metrics_endpoint_is_not_rate_limited(client):
    """The /metrics endpoint must never return 429 (not a business endpoint)."""
    import rag_server as srv
    from unittest.mock import AsyncMock

    # Simulate Redis over-limit
    over_limit_redis = AsyncMock()
    over_limit_redis.incr = AsyncMock(return_value=9999)
    over_limit_redis.expire = AsyncMock(return_value=True)
    over_limit_redis.ping = AsyncMock(return_value=b"PONG")
    over_limit_redis.get = AsyncMock(return_value=None)
    srv.app.state.redis = over_limit_redis

    resp = await client.get("/metrics")
    assert resp.status_code != 429
