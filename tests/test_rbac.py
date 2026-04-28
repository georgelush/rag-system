"""
tests/test_rbac.py — Tests for namespace-level RBAC (JWT tokens + POST /v1/tokens).
"""

import os
import pytest
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_API_KEY


# ── Helpers ───────────────────────────────────────────────────────────────────

TEST_RBAC_SECRET = "test-rbac-secret-for-pytest-long-enough-32b"
TEST_TENANT = "tenant-test"
TEST_NS = "ns"


def _mint(namespaces: dict, expires_in: int = 3600) -> str:
    """Mint a JWT using rag.rbac.mint_token with the test secret."""
    with (
        patch("rag.rbac.RBAC_ENABLED", True),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        from rag.rbac import mint_token
        token, _ = mint_token(TEST_TENANT, namespaces, expires_in)
    return token


def _auth_headers_with_jwt(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-Request-ID": "req-rbac-test",
        "X-Tenant-ID":  TEST_TENANT,
    }


# ── AuthResult: master key always has full access ─────────────────────────────

def test_authresult_admin_can_do_everything():
    from rag.rbac import AuthResult
    auth = AuthResult(request_id="r", tenant_id="t", is_admin=True)
    assert auth.can("any-ns", "read")
    assert auth.can("any-ns", "write")
    assert auth.can("any-ns", "admin")


# ── AuthResult: scoped token permission logic ─────────────────────────────────

def test_authresult_read_token_allows_read_only():
    from rag.rbac import AuthResult
    auth = AuthResult(
        request_id="r", tenant_id="t", is_admin=False,
        _ns_perms={"my-ns": {"read"}},
    )
    assert auth.can("my-ns", "read")
    assert not auth.can("my-ns", "write")
    assert not auth.can("my-ns", "admin")


def test_authresult_wildcard_grants_all_namespaces():
    from rag.rbac import AuthResult
    auth = AuthResult(
        request_id="r", tenant_id="t", is_admin=False,
        _ns_perms={"*": {"read"}},
    )
    assert auth.can("ns-a", "read")
    assert auth.can("ns-b", "read")
    assert not auth.can("ns-a", "write")


def test_authresult_write_includes_read():
    """write permission level (1) >= read (0) — so write grants read too."""
    from rag.rbac import AuthResult
    auth = AuthResult(
        request_id="r", tenant_id="t", is_admin=False,
        _ns_perms={"my-ns": {"write"}},
    )
    assert auth.can("my-ns", "read")
    assert auth.can("my-ns", "write")
    assert not auth.can("my-ns", "admin")


def test_authresult_assert_forbidden_raises_403():
    from fastapi import HTTPException
    from rag.rbac import AuthResult
    auth = AuthResult(
        request_id="r", tenant_id="t", is_admin=False,
        _ns_perms={"my-ns": {"read"}},
    )
    with pytest.raises(HTTPException) as exc_info:
        auth.assert_can_write("my-ns")
    assert exc_info.value.status_code == 403


# ── JWT decode ────────────────────────────────────────────────────────────────

def test_decode_returns_none_when_rbac_disabled():
    from rag.rbac import decode_namespace_token
    with patch("rag.rbac.RBAC_ENABLED", False):
        result = decode_namespace_token("any-token", "tenant", "req-1")
    assert result is None


def test_decode_valid_token():
    token = _mint({"my-ns": ["read", "write"]})
    with (
        patch("rag.rbac.RBAC_ENABLED", True),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        from rag.rbac import decode_namespace_token
        auth = decode_namespace_token(token, TEST_TENANT, "req-1")
    assert auth is not None
    assert auth.tenant_id == TEST_TENANT
    assert not auth.is_admin
    assert auth.can("my-ns", "read")
    assert auth.can("my-ns", "write")
    assert not auth.can("my-ns", "admin")


def test_decode_expired_token_raises_401():
    """Expired JWT should raise HTTPException 401."""
    token = _mint({"ns": ["read"]}, expires_in=-1)  # already expired
    from fastapi import HTTPException
    with (
        patch("rag.rbac.RBAC_ENABLED", True),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        from rag.rbac import decode_namespace_token
        with pytest.raises(HTTPException) as exc_info:
            decode_namespace_token(token, TEST_TENANT, "req-1")
    assert exc_info.value.status_code == 401
    assert "expired" in str(exc_info.value.detail).lower()


def test_decode_wrong_tenant_raises_401():
    token = _mint({"ns": ["read"]})
    from fastapi import HTTPException
    with (
        patch("rag.rbac.RBAC_ENABLED", True),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        from rag.rbac import decode_namespace_token
        with pytest.raises(HTTPException) as exc_info:
            decode_namespace_token(token, "different-tenant", "req-1")
    assert exc_info.value.status_code == 401


# ── POST /v1/tokens endpoint ──────────────────────────────────────────────────

async def test_create_token_requires_master_key(client: AsyncClient):
    """Using a JWT (not master key) to call POST /v1/tokens should return 403."""
    token = _mint({"ns": ["read"]})
    with (
        patch("rag.rbac.RBAC_ENABLED", True),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
        patch("rag.routers.tokens_router.RBAC_ENABLED", True),
        patch("rag.routers.tokens_router.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        resp = await client.post(
            "/v1/tokens",
            json={"tenant_id": TEST_TENANT, "namespaces": {"ns": ["read"]}},
            headers=_auth_headers_with_jwt(token),
        )
    assert resp.status_code in (401, 403)


async def test_create_token_with_rbac_disabled_returns_400(client: AsyncClient):
    """When RBAC is disabled, POST /v1/tokens returns 400."""
    with patch("rag.routers.tokens_router.RBAC_ENABLED", False):
        resp = await client.post(
            "/v1/tokens",
            json={"tenant_id": TEST_TENANT, "namespaces": {"ns": ["read"]}},
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 400
    assert "RBAC" in resp.json()["error"]["message"]


async def test_create_token_success(client: AsyncClient):
    """Master key + RBAC enabled → returns a JWT token."""
    with (
        patch("rag.routers.tokens_router.RBAC_ENABLED", True),
        patch("rag.routers.tokens_router.RBAC_SECRET", TEST_RBAC_SECRET),
        patch("rag.rbac.RBAC_SECRET", TEST_RBAC_SECRET),
    ):
        resp = await client.post(
            "/v1/tokens",
            json={
                "tenant_id":          TEST_TENANT,
                "namespaces":         {"ns": ["read"]},
                "expires_in_seconds": 3600,
            },
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 201
    data = resp.json()
    assert "token" in data
    assert data["tenant_id"] == TEST_TENANT
    assert "expires_at" in data
    # Token must be a non-empty string (basic JWT sanity check)
    parts = data["token"].split(".")
    assert len(parts) == 3
