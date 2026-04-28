import os
import time
from contextlib import asynccontextmanager

# ── Env vars — must come before any rag import ─────────────────────────────────
os.environ["RAG_API_KEY"] = "test-api-key-12345"
os.environ.setdefault("LLM_API_KEY", "sk-test")
os.environ.setdefault("LLM_PROXY", "")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport

# ── Constants used across tests ────────────────────────────────────────────────

TEST_API_KEY = "test-api-key-12345"

AUTH_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {TEST_API_KEY}",
    "X-Request-ID": "req-test-001",
    "X-Tenant-ID":  "tenant-test",
}


# ── Mock factories ─────────────────────────────────────────────────────────────

def make_mock_redis() -> AsyncMock:
    r = AsyncMock()
    r.ping.return_value = b"PONG"
    r.get.return_value = None
    r.set.return_value = True
    r.setex.return_value = True
    r.aclose = AsyncMock()
    return r


def make_mock_qdrant() -> AsyncMock:
    q = AsyncMock()
    collections = MagicMock()
    collections.collections = []
    q.get_collections.return_value = collections
    q.close = AsyncMock()
    return q


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis() -> AsyncMock:
    return make_mock_redis()


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    return make_mock_qdrant()


@pytest_asyncio.fixture
async def client(mock_redis: AsyncMock, mock_qdrant: AsyncMock) -> AsyncClient:
    """FastAPI test client with mocked Redis and Qdrant.

    Sets app.state directly and replaces lifespan with a no-op so no real
    network connections are attempted regardless of how ASGITransport handles
    the lifespan startup event.
    """
    import rag_server as srv

    # Inject mocks directly into app state (works even if lifespan isn't triggered)
    srv.app.state.qdrant = mock_qdrant
    srv.app.state.redis = mock_redis
    srv.app.state.start_time = time.time()

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield  # state already set above

    original_lifespan = srv.app.router.lifespan_context
    srv.app.router.lifespan_context = _noop_lifespan
    try:
        async with AsyncClient(
            transport=ASGITransport(app=srv.app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        srv.app.router.lifespan_context = original_lifespan
