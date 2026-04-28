"""
tests/test_batch_query.py — Integration tests for POST /v1/query/batch.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import QueryResponse, Chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key-12345",
    "X-Request-ID": "req-batch-001",
    "X-Tenant-ID": "tenant-test",
}

_USAGE = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "model_id": "test"}

_EMPTY_QUERY_RESPONSE = QueryResponse(
    answer="",
    request_id="req-batch-001",
    usage=_USAGE,
    latency_ms=0,
    model_version="test",
)


def _make_query_payload(namespaces=None, question="What is X?"):
    return {"namespaces": namespaces or ["ns-a"], "question": question}


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_query_requires_auth(client):
    resp = await client.post(
        "/v1/query/batch",
        json={"queries": [_make_query_payload()]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_batch_query_empty_list_rejected(client):
    resp = await client.post(
        "/v1/query/batch",
        json={"queries": []},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_batch_query_single_success(client):
    with patch("rag.routers.query_router.run_query", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = _EMPTY_QUERY_RESPONSE
        resp = await client.post(
            "/v1/query/batch",
            json={"queries": [_make_query_payload()]},
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 1
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_query_multiple_parallel(client):
    call_order = []

    async def mock_run(qdrant, req, request_id, tenant_id, redis=None):
        call_order.append(req.namespaces)
        return _EMPTY_QUERY_RESPONSE

    with patch("rag.routers.query_router.run_query", side_effect=mock_run):
        resp = await client.post(
            "/v1/query/batch",
            json={
                "queries": [
                    _make_query_payload(["ns-a"], "Q1"),
                    _make_query_payload(["ns-b"], "Q2"),
                    _make_query_payload(["ns-c"], "Q3"),
                ]
            },
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3


@pytest.mark.asyncio
async def test_batch_query_exceeds_max_returns_422(client):
    """More than BATCH_QUERY_MAX (10) queries → 422."""
    queries = [_make_query_payload() for _ in range(11)]
    resp = await client.post(
        "/v1/query/batch",
        json={"queries": queries},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_batch_query_per_query_error_is_inline(client):
    """A run_query exception for one item returns inline error, not 500."""

    async def mock_run(qdrant, req, request_id, tenant_id, redis=None):
        if req.namespaces == ["ns-fail"]:
            raise ValueError("boom")
        return _EMPTY_QUERY_RESPONSE

    with patch("rag.routers.query_router.run_query", side_effect=mock_run):
        resp = await client.post(
            "/v1/query/batch",
            json={
                "queries": [
                    _make_query_payload(["ns-a"]),
                    _make_query_payload(["ns-fail"]),
                ]
            },
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert "error" in results[1]
    assert results[1]["error"]["code"] == "internal_error"
