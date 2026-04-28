"""
rag/rate_limiter.py — Per-tenant fixed-window rate limiter.

Uses Redis INCR + EXPIRE — atomic, distributed, no extra dependencies.

Default limits (requests per minute per tenant):
  /v1/query  → RATE_LIMIT_QUERY   (default 60 — 1 req/s)
  /v1/ingest → RATE_LIMIT_INGEST  (default 20)
  all others → RATE_LIMIT_DEFAULT (default 120)

Returns HTTP 429 with Retry-After header on breach.
Fails OPEN if Redis is unreachable — never blocks legitimate traffic.
"""

import logging
import math
import os
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

log = logging.getLogger(__name__)

_LIMITS: dict[str, int] = {
    "/v1/query":  int(os.environ.get("RATE_LIMIT_QUERY",   "60")),
    "/v1/ingest": int(os.environ.get("RATE_LIMIT_INGEST",  "20")),
}
_DEFAULT_LIMIT = int(os.environ.get("RATE_LIMIT_DEFAULT", "120"))
_WINDOW_SEC    = 60  # fixed window size in seconds

# Paths that bypass rate limiting entirely
_EXEMPT = ("/v1/health", "/v1/openapi.json", "/docs", "/redoc", "/openapi.json")


def _limit_for(path: str) -> int:
    for prefix, limit in _LIMITS.items():
        if path.startswith(prefix):
            return limit
    return _DEFAULT_LIMIT


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter middleware.

    Attaches to the FastAPI app via:
        app.add_middleware(RateLimiterMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip exempt paths (health, openapi, etc.)
        if any(path.startswith(e) for e in _EXEMPT):
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID")
        if not tenant_id:
            # Unauthenticated requests — let auth middleware handle 401
            return await call_next(request)

        limit  = _limit_for(path)
        window = int(time.time() / _WINDOW_SEC)
        key    = f"rag:rl:{tenant_id}:{path}:{window}"

        try:
            redis = request.app.state.redis
            count = await redis.incr(key)
            if count == 1:
                # Set expiry on first increment (2× window for safety margin)
                await redis.expire(key, _WINDOW_SEC * 2)

            if count > limit:
                retry_after = math.ceil(_WINDOW_SEC - (time.time() % _WINDOW_SEC))
                log.warning(
                    "Rate limit exceeded",
                    extra={
                        "tenant_id": tenant_id,
                        "path":      path,
                        "count":     count,
                        "limit":     limit,
                    },
                )
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "error": {
                            "code":       "rate_limited",
                            "message":    (
                                f"Too many requests. Limit: {limit}/min. "
                                f"Retry after {retry_after}s."
                            ),
                            "request_id": request.headers.get("X-Request-ID", ""),
                        }
                    },
                )
        except AttributeError:
            # app.state.redis not available during startup — skip
            pass
        except Exception as exc:
            # Redis error — fail OPEN (never block traffic)
            log.warning("Rate limiter Redis error — skipping", extra={"error": str(exc)})

        return await call_next(request)
