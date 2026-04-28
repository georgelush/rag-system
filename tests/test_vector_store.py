"""
tests/test_vector_store.py — Unit tests for rag/vector_store.py.

Covers:
  - collection_name returns "{tenant}__{namespace}"
  - Different tenants get different collection names (isolation)
  - Same tenant, different namespaces → different collection names
  - _is_hybrid returns True when sparse_vectors_config is set
  - _is_hybrid returns False for collection without sparse vectors (legacy)
  - search_chunks uses dense-only path for non-hybrid collections
  - search_chunks uses hybrid path for hybrid collections
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── collection_name ────────────────────────────────────────────────────────────

def test_collection_name_format():
    from rag.vector_store import collection_name
    assert collection_name("tenant-a", "ns-1") == "tenant-a__ns-1"


def test_collection_name_different_tenants_are_isolated():
    from rag.vector_store import collection_name
    assert collection_name("tenant-a", "ns-1") != collection_name("tenant-b", "ns-1")


def test_collection_name_different_namespaces_within_same_tenant():
    from rag.vector_store import collection_name
    assert collection_name("tenant-a", "ns-1") != collection_name("tenant-a", "ns-2")


def test_collection_name_same_inputs_are_equal():
    from rag.vector_store import collection_name
    assert collection_name("t", "n") == collection_name("t", "n")


def test_collection_name_uses_double_underscore_separator():
    from rag.vector_store import collection_name
    name = collection_name("my-tenant", "my-namespace")
    assert "__" in name
    parts = name.split("__")
    assert parts[0] == "my-tenant"
    assert parts[1] == "my-namespace"


# ── _is_hybrid ─────────────────────────────────────────────────────────────────

def _make_qdrant_client(has_sparse: bool, col_name: str = "tenant-1__ns-1") -> AsyncMock:
    """Build a mock AsyncQdrantClient whose get_collection returns appropriate config."""
    sparse_config = MagicMock() if has_sparse else None

    params = MagicMock()
    params.sparse_vectors_config = sparse_config

    config = MagicMock()
    config.params = params

    collection_info = MagicMock()
    collection_info.config = config

    # Mock get_collection (used by _is_hybrid)
    client = AsyncMock()
    client.get_collection = AsyncMock(return_value=collection_info)

    # Mock get_collections (used by search_chunks existence check)
    col_stub = MagicMock()
    col_stub.name = col_name
    collections_result = MagicMock()
    collections_result.collections = [col_stub]
    client.get_collections = AsyncMock(return_value=collections_result)

    return client


@pytest.mark.asyncio
async def test_is_hybrid_returns_true_when_sparse_config_present():
    from rag.vector_store import _is_hybrid
    client = _make_qdrant_client(has_sparse=True)
    result = await _is_hybrid(client, "tenant-1__ns-1")
    assert result is True


@pytest.mark.asyncio
async def test_is_hybrid_returns_false_when_sparse_config_absent():
    from rag.vector_store import _is_hybrid
    client = _make_qdrant_client(has_sparse=False)
    result = await _is_hybrid(client, "tenant-1__ns-1")
    assert result is False


@pytest.mark.asyncio
async def test_is_hybrid_returns_false_on_qdrant_exception():
    """If Qdrant raises (e.g. collection not found), _is_hybrid should return False."""
    from rag.vector_store import _is_hybrid

    client = AsyncMock()
    client.get_collection = AsyncMock(side_effect=Exception("not found"))
    result = await _is_hybrid(client, "missing-collection")
    assert result is False


# ── search_chunks routing ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_chunks_uses_dense_path_for_non_hybrid_collection():
    """For legacy collections (no sparse), _search_dense must be called."""
    from rag import vector_store as vs
    from unittest.mock import patch, AsyncMock

    client = _make_qdrant_client(has_sparse=False)
    mock_chunks = [MagicMock()]

    with patch.object(vs, "_search_dense", new_callable=AsyncMock, return_value=mock_chunks) as mock_dense, \
         patch.object(vs, "_search_hybrid", new_callable=AsyncMock, return_value=[]) as mock_hybrid:

        result = await vs.search_chunks(
            client=client,
            col="tenant-1__ns-1",
            dense_vector=[0.1, 0.2],
            top_k=5,
            sparse_vector=None,
        )

    mock_dense.assert_awaited_once()
    mock_hybrid.assert_not_awaited()
    assert result == mock_chunks


@pytest.mark.asyncio
async def test_search_chunks_uses_hybrid_path_for_hybrid_collection():
    """For hybrid collections (with sparse), _search_hybrid must be called."""
    from rag import vector_store as vs
    from unittest.mock import patch, AsyncMock

    client = _make_qdrant_client(has_sparse=True)
    mock_chunks = [MagicMock()]

    with patch.object(vs, "_search_hybrid", new_callable=AsyncMock, return_value=mock_chunks) as mock_hybrid, \
         patch.object(vs, "_search_dense", new_callable=AsyncMock, return_value=[]) as mock_dense:

        result = await vs.search_chunks(
            client=client,
            col="tenant-1__ns-1",
            dense_vector=[0.1, 0.2],
            top_k=5,
            sparse_vector={"indices": [1, 2], "values": [0.5, 0.3]},
        )

    mock_hybrid.assert_awaited_once()
    mock_dense.assert_not_awaited()
    assert result == mock_chunks
