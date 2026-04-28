"""
tests/test_health.py — Health endpoint integration tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import AUTH_HEADERS


async def test_health_ok_returns_200(client, mock_qdrant, mock_redis):
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mock_redis.ping.return_value = b"PONG"

    resp = await client.get("/v1/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["dependencies"]["qdrant"] == "ok"
    assert data["dependencies"]["redis"] == "ok"


async def test_health_qdrant_down_returns_503(client, mock_qdrant, mock_redis):
    mock_qdrant.get_collections.side_effect = Exception("Connection refused")
    mock_redis.ping.return_value = b"PONG"

    resp = await client.get("/v1/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "down"
    assert data["dependencies"]["qdrant"] == "down"
    assert data["dependencies"]["redis"] == "ok"


async def test_health_redis_down_returns_503(client, mock_qdrant, mock_redis):
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mock_redis.ping.side_effect = Exception("Redis unavailable")

    resp = await client.get("/v1/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "down"
    assert data["dependencies"]["redis"] == "down"


async def test_health_response_schema(client, mock_qdrant, mock_redis):
    """Health response must include required fields."""
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mock_redis.ping.return_value = b"PONG"

    resp = await client.get("/v1/health")
    data = resp.json()

    assert "status" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "dependencies" in data
    assert isinstance(data["uptime_seconds"], int)


async def test_health_does_not_require_auth(client):
    """Health endpoint must be accessible without Authorization header."""
    resp = await client.get("/v1/health")
    # Should not return 401
    assert resp.status_code != 401
