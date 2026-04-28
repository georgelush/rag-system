"""
src/rag/auth.py — Bearer token validation and required headers check.

Spec §2: every request MUST carry Authorization, X-Request-ID, X-Tenant-ID.
Returns 401 on bad/missing key, 400 on missing headers.

Auth flow
---------
1. Master API key (RAG_API_KEY) → AuthResult(is_admin=True) — full access
2. JWT namespace token (RBAC_ENABLED=true) → AuthResult with scoped perms
3. Neither → 401

Usage: inject ``require_auth`` as a FastAPI dependency in any protected router.
"""

import hmac
import os
import uuid
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException

from rag.models import ErrorCode, ErrorDetail, ErrorResponse
from rag.rbac import AuthResult, decode_namespace_token

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

RAG_API_KEY: str = os.environ.get("RAG_API_KEY", "")

if not RAG_API_KEY:
    raise RuntimeError("RAG_API_KEY env var is required but not set.")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_error(code: ErrorCode, message: str, request_id: str, status: int) -> HTTPException:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    return HTTPException(status_code=status, detail=body.model_dump(mode="json"))

# ── Dependency ─────────────────────────────────────────────────────────────────

async def require_auth(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_request_id:  Annotated[str | None, Header(alias="X-Request-ID")]  = None,
    x_tenant_id:   Annotated[str | None, Header(alias="X-Tenant-ID")]   = None,
) -> AuthResult:
    """
    FastAPI dependency injected into all authenticated routers.

    Returns an AuthResult on success.  The caller can inspect
    ``auth.is_admin`` or call ``auth.assert_can_read/write/admin(ns)``
    to enforce namespace-level permissions.
    """
    # Fallback request_id for error responses before we validate it
    req_id = x_request_id or str(uuid.uuid4())

    if not authorization or not authorization.startswith("Bearer "):
        raise _make_error(
            ErrorCode.unauthorized,
            "Missing or malformed Authorization header.",
            req_id, 401,
        )

    if not x_request_id:
        raise _make_error(
            ErrorCode.invalid_request,
            "Missing required header: X-Request-ID",
            req_id, 400,
        )

    if not x_tenant_id:
        raise _make_error(
            ErrorCode.invalid_request,
            "Missing required header: X-Tenant-ID",
            req_id, 400,
        )

    token = authorization.removeprefix("Bearer ").strip()

    # ── 1. Try master API key ─────────────────────────────────────────────────
    if hmac.compare_digest(token, RAG_API_KEY):
        return AuthResult(
            request_id=req_id,
            tenant_id=x_tenant_id,
            is_admin=True,
        )

    # ── 2. Try JWT namespace token (only when RBAC is enabled) ───────────────
    result = decode_namespace_token(token, x_tenant_id, req_id)
    if result is not None:
        return result

    # ── 3. Neither matched → 401 ─────────────────────────────────────────────
    raise _make_error(
        ErrorCode.unauthorized,
        "Invalid API key.",
        req_id, 401,
    )
