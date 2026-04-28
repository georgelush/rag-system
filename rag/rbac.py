"""
rag/rbac.py — Namespace-level RBAC via signed JWT tokens.

When RBAC_ENABLED=true (default: false), callers can authenticate with
namespace-scoped JWT tokens in addition to the master API key.

Token format (HS256, signed with RBAC_SECRET)
----------------------------------------------
    {
        "sub":        "<uuid>",
        "tenant_id":  "acme",
        "namespaces": {
            "contracts": ["read", "write"],
            "public":    ["read"],
            "*":         ["read"]     # wildcard: all namespaces
        },
        "iat": 1714257600,
        "exp": 1714261200
    }

Permission hierarchy
--------------------
    read < write < admin

    read   — POST /v1/query, POST /v1/query/stream,
             GET  /v1/namespaces/{ns}/stats,
             GET  /v1/ingest/{job_id}
    write  — POST /v1/ingest,
             DELETE /v1/namespaces/{ns}/sources/{src}
    admin  — DELETE /v1/namespaces/{ns}

    POST /v1/tokens is always restricted to master API key (is_admin=True).

Backward compatibility
----------------------
When RBAC_ENABLED=false (the default), ``decode_namespace_token`` returns
None immediately and the master API key is the only accepted credential.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RBAC_ENABLED: bool = os.environ.get("RBAC_ENABLED", "false").lower() == "true"
RBAC_SECRET: str   = os.environ.get("RBAC_SECRET", "")

Permission = Literal["read", "write", "admin"]
_LEVEL: dict[str, int] = {"read": 0, "write": 1, "admin": 2}


# ── AuthResult ────────────────────────────────────────────────────────────────

@dataclass
class AuthResult:
    """Returned by ``require_auth`` after successful authentication."""

    request_id: str
    tenant_id:  str
    is_admin:   bool               = True
    _ns_perms:  dict[str, set[str]] = field(default_factory=dict)

    # ── Capability checks ─────────────────────────────────────────────────────

    def can(self, namespace_id: str, action: Permission) -> bool:
        """Return True when this context grants *action* on *namespace_id*."""
        if self.is_admin:
            return True
        needed = _LEVEL[action]
        grants = (
            self._ns_perms.get("*", set()) | self._ns_perms.get(namespace_id, set())
        )
        return any(_LEVEL.get(p, -1) >= needed for p in grants)

    def assert_can_read(self, namespace_id: str) -> None:
        if not self.can(namespace_id, "read"):
            _forbidden(self.request_id, namespace_id, "read")

    def assert_can_write(self, namespace_id: str) -> None:
        if not self.can(namespace_id, "write"):
            _forbidden(self.request_id, namespace_id, "write")

    def assert_can_admin(self, namespace_id: str) -> None:
        if not self.can(namespace_id, "admin"):
            _forbidden(self.request_id, namespace_id, "admin")


def _forbidden(request_id: str, namespace_id: str, action: str) -> None:
    from fastapi import HTTPException
    raise HTTPException(
        status_code=403,
        detail={
            "error": {
                "code":       "forbidden",
                "message":    (
                    f"Token does not grant '{action}' access "
                    f"to namespace '{namespace_id}'."
                ),
                "request_id": request_id,
            }
        },
    )


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _jwt():
    """Lazy-import PyJWT to avoid hard dependency when RBAC is disabled."""
    try:
        import jwt  # noqa: PLC0415
        return jwt
    except ImportError as exc:
        raise RuntimeError(
            "PyJWT is required when RBAC_ENABLED=true. "
            "Run: pip install 'PyJWT>=2.0.0,<3.0.0'"
        ) from exc


def decode_namespace_token(
    raw_token: str,
    tenant_id: str,
    request_id: str,
) -> AuthResult | None:
    """Try to decode *raw_token* as a JWT namespace token.

    Returns:
        AuthResult — on success (JWT is valid, not expired, tenant matches)
        None       — token is not a valid JWT; caller should try master key

    Raises:
        HTTPException 401 — JWT is structurally valid but expired / wrong tenant
    """
    if not RBAC_ENABLED or not RBAC_SECRET:
        return None

    jwt = _jwt()
    try:
        payload = jwt.decode(raw_token, RBAC_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code":       "unauthorized",
                    "message":    "Namespace token has expired.",
                    "request_id": request_id,
                }
            },
        )
    except Exception:
        # Not a JWT or invalid signature — let caller try master key
        return None

    jwt_tenant = payload.get("tenant_id", "")
    if jwt_tenant != tenant_id:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code":       "unauthorized",
                    "message":    (
                        "Token tenant_id does not match X-Tenant-ID header."
                    ),
                    "request_id": request_id,
                }
            },
        )

    ns_perms_raw: dict = payload.get("namespaces", {})
    ns_perms = {ns: set(perms) for ns, perms in ns_perms_raw.items()}

    return AuthResult(
        request_id=request_id,
        tenant_id=tenant_id,
        is_admin=False,
        _ns_perms=ns_perms,
    )


def mint_token(
    tenant_id: str,
    namespaces: dict[str, list[str]],
    expires_in_seconds: int = 3600,
) -> tuple[str, datetime]:
    """Sign and return a JWT namespace token.

    Returns:
        (jwt_string, expires_at_utc_datetime)

    Raises:
        RuntimeError — if RBAC_SECRET is not set
    """
    if not RBAC_SECRET:
        raise RuntimeError(
            "RBAC_SECRET must be set to mint namespace tokens."
        )

    jwt = _jwt()
    now = datetime.now(timezone.utc)
    exp = datetime.fromtimestamp(
        now.timestamp() + expires_in_seconds, tz=timezone.utc
    )
    payload = {
        "sub":        str(uuid.uuid4()),
        "tenant_id":  tenant_id,
        "namespaces": namespaces,
        "iat":        int(now.timestamp()),
        "exp":        int(exp.timestamp()),
    }
    token: str = jwt.encode(payload, RBAC_SECRET, algorithm="HS256")
    return token, exp
