"""
tests/test_auth.py — Auth dependency: bearer token, required headers, timing safety.
"""

import hmac
import inspect

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_API_KEY

# ── Unit: timing-safe compare ──────────────────────────────────────────────────

def test_uses_hmac_compare_digest():
    """P0 fix: verify timing-attack mitigation is in place."""
    import rag.auth as auth_mod
    src = inspect.getsource(auth_mod.require_auth)
    assert "hmac.compare_digest" in src, "require_auth must use hmac.compare_digest"


# ── Integration: missing / malformed headers ───────────────────────────────────

async def test_missing_authorization_returns_401(client: AsyncClient):
    headers = {k: v for k, v in AUTH_HEADERS.items() if k != "Authorization"}
    resp = await client.post("/v1/query", json={"question": "x", "namespaces": ["ns"]}, headers=headers)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_malformed_bearer_returns_401(client: AsyncClient):
    headers = {**AUTH_HEADERS, "Authorization": "Token abc123"}
    resp = await client.post("/v1/query", json={"question": "x", "namespaces": ["ns"]}, headers=headers)
    assert resp.status_code == 401


async def test_wrong_token_returns_401(client: AsyncClient):
    headers = {**AUTH_HEADERS, "Authorization": "Bearer wrong-key"}
    resp = await client.post("/v1/query", json={"question": "x", "namespaces": ["ns"]}, headers=headers)
    assert resp.status_code == 401


async def test_missing_request_id_returns_400(client: AsyncClient):
    headers = {k: v for k, v in AUTH_HEADERS.items() if k != "X-Request-ID"}
    resp = await client.post("/v1/query", json={"question": "x", "namespaces": ["ns"]}, headers=headers)
    assert resp.status_code == 400
    assert "X-Request-ID" in resp.json()["error"]["message"]


async def test_missing_tenant_id_returns_400(client: AsyncClient):
    headers = {k: v for k, v in AUTH_HEADERS.items() if k != "X-Tenant-ID"}
    resp = await client.post("/v1/query", json={"question": "x", "namespaces": ["ns"]}, headers=headers)
    assert resp.status_code == 400
    assert "X-Tenant-ID" in resp.json()["error"]["message"]


async def test_valid_auth_reaches_handler(client: AsyncClient, mock_qdrant):
    """Valid credentials should not be blocked — endpoint returns 200 or pipeline error."""
    from unittest.mock import AsyncMock, patch

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        resp = await client.post(
            "/v1/query",
            json={"question": "What is X?", "namespaces": ["ns"], "include_answer": False},
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 200
