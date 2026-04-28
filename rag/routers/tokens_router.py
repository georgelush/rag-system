"""rag/routers/tokens_router.py — POST /v1/tokens (namespace token minting)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from rag.auth import require_auth
from rag.rbac import AuthResult, RBAC_ENABLED, RBAC_SECRET, mint_token

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class TokenRequest(BaseModel):
    tenant_id:          str                      = Field(...)
    namespaces:         dict[str, list[str]]      = Field(
        ...,
        description=(
            "Map of namespace_id → list of permissions. "
            "Use '*' as a wildcard to grant access to all namespaces. "
            "Valid permissions: 'read', 'write', 'admin'."
        ),
        examples=[{"contracts": ["read", "write"], "*": ["read"]}],
    )
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400 * 30)


class TokenResponse(BaseModel):
    token:      str
    tenant_id:  str
    expires_at: str  # ISO 8601 UTC


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/v1/tokens", response_model=TokenResponse, status_code=201)
async def create_token(
    body: TokenRequest,
    request: Request,
    auth: Annotated[AuthResult, Depends(require_auth)],
) -> TokenResponse:
    """Mint a signed JWT namespace token.

    **Requires master API key** — namespace-scoped JWT tokens cannot be used
    to mint further tokens (privilege escalation guard).

    The minted token can be used in ``Authorization: Bearer <token>`` headers
    on any endpoint.  It carries per-namespace read/write/admin claims and
    honours the ``expires_in_seconds`` TTL.

    RBAC must be enabled server-side (``RBAC_ENABLED=true``) and
    ``RBAC_SECRET`` must be set.
    """
    if not auth.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code":       "forbidden",
                    "message":    (
                        "Only master API key holders can mint namespace tokens."
                    ),
                    "request_id": auth.request_id,
                }
            },
        )

    if not RBAC_ENABLED:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code":       "invalid_request",
                    "message":    (
                        "RBAC is disabled on this server. "
                        "Set RBAC_ENABLED=true and RBAC_SECRET to use namespace tokens."
                    ),
                    "request_id": auth.request_id,
                }
            },
        )

    if not RBAC_SECRET:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code":       "internal_error",
                    "message":    "RBAC_SECRET is not configured on the server.",
                    "request_id": auth.request_id,
                }
            },
        )

    token, expires_at = mint_token(
        tenant_id=body.tenant_id,
        namespaces=body.namespaces,
        expires_in_seconds=body.expires_in_seconds,
    )

    return TokenResponse(
        token=token,
        tenant_id=body.tenant_id,
        expires_at=expires_at.isoformat(),
    )
