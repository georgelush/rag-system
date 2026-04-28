"""
tests/test_migrate.py — Unit tests for rag/migrate.py (embedding migration CLI).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_qdrant(
    collection_names: list[str] = (),
    point_payloads: list[dict] | None = None,
    points_count: int = 0,
):
    """Return a mock AsyncQdrantClient pre-configured for migration tests."""
    qdrant = AsyncMock()

    # get_collections → fixed list
    collections_result = MagicMock()
    collections_result.collections = [
        MagicMock(name=n) for n in collection_names
    ]
    # Each MagicMock needs .name to equal the string
    for mock_col, name in zip(
        collections_result.collections, collection_names
    ):
        mock_col.name = name
    qdrant.get_collections.return_value = collections_result

    # scroll → one batch, then stop
    records = []
    if point_payloads:
        for i, payload in enumerate(point_payloads):
            pt = MagicMock()
            pt.id = str(i + 1)
            pt.payload = payload
            pt.vector = {"dense": [0.1] * 1536, "sparse": MagicMock()}
            records.append(pt)

    # scroll with (result, None) to signal end
    qdrant.scroll.return_value = (records, None)
    qdrant.upsert.return_value = None
    qdrant.create_collection.return_value = None
    qdrant.delete_collection.return_value = None

    # get_collection → points_count for verification
    col_info = MagicMock()
    col_info.points_count = points_count if points_count else len(records)
    qdrant.get_collection.return_value = col_info
    qdrant.close.return_value = None
    return qdrant


def _make_openai(embedding_dim: int = 3072):
    """Return a mock AsyncOpenAI that returns zero-vectors."""
    openai = AsyncMock()
    embed_result = MagicMock()

    async def _create(**kwargs):
        n = len(kwargs.get("input", []))
        data = [MagicMock(embedding=[0.0] * embedding_dim) for _ in range(n)]
        result = MagicMock()
        result.data = data
        return result

    openai.embeddings.create.side_effect = _create
    return openai


# ── C1: same-model short-circuit ─────────────────────────────────────────────

async def test_same_model_returns_zero_immediately():
    from rag.migrate import run_migration
    result = await run_migration(
        "tenant", "ns",
        "text-embedding-3-small", "text-embedding-3-small",
    )
    assert result == 0


# ── C1: invalid model validation ─────────────────────────────────────────────

async def test_unsupported_source_model_raises_value_error():
    from rag.migrate import run_migration
    with pytest.raises(ValueError, match="Unsupported source model"):
        await run_migration("t", "ns", "bad-model", "text-embedding-3-large")


async def test_unsupported_target_model_raises_value_error():
    from rag.migrate import run_migration
    with pytest.raises(ValueError, match="Unsupported target model"):
        await run_migration("t", "ns", "text-embedding-3-small", "bad-model")


# ── C1: missing source collection ────────────────────────────────────────────

async def test_missing_source_collection_raises_runtime_error():
    from rag.migrate import run_migration
    qdrant = _make_qdrant(collection_names=[])  # source not present
    with pytest.raises(RuntimeError, match="not found"):
        await run_migration(
            "tenant", "ns",
            "text-embedding-3-small", "text-embedding-3-large",
            _qdrant=qdrant,
            _openai=_make_openai(),
        )


# ── C1: empty collection ──────────────────────────────────────────────────────

async def test_empty_collection_returns_zero():
    from rag.migrate import run_migration
    qdrant = _make_qdrant(
        collection_names=["tenant__ns"],
        point_payloads=[],  # empty
    )
    col_info = MagicMock()
    col_info.points_count = 0
    qdrant.get_collection.return_value = col_info

    result = await run_migration(
        "tenant", "ns",
        "text-embedding-3-small", "text-embedding-3-large",
        _qdrant=qdrant,
        _openai=_make_openai(),
    )
    assert result == 0
    # Should not create any collections
    qdrant.create_collection.assert_not_called()


# ── C1: happy-path migration ──────────────────────────────────────────────────

async def test_full_migration_happy_path():
    from rag.migrate import run_migration

    payloads = [
        {"content": f"chunk {i}", "source_id": "src", "namespace_id": "ns"}
        for i in range(3)
    ]
    qdrant = _make_qdrant(
        collection_names=["tenant__ns"],
        point_payloads=payloads,
    )
    openai = _make_openai(embedding_dim=3072)

    with patch("rag.migrate.encode_sparse", return_value=MagicMock()):
        result = await run_migration(
            "tenant", "ns",
            "text-embedding-3-small", "text-embedding-3-large",
            _qdrant=qdrant,
            _openai=openai,
        )

    assert result == 3
    # Temp + final source collections created
    assert qdrant.create_collection.call_count == 2
    # Source + temp deleted (2 deletes)
    assert qdrant.delete_collection.call_count == 2


# ── C1: resume after interruption ────────────────────────────────────────────

async def test_resume_when_temp_exists_source_gone():
    """If temp collection exists but source is gone, migration resumes from copy step."""
    from rag.migrate import run_migration

    payloads = [{"content": "chunk", "source_id": "s", "namespace_id": "ns"}]
    # Only temp collection exists (source already deleted in prior run)
    qdrant = _make_qdrant(
        collection_names=["tenant__ns__migration_tmp"],
        point_payloads=payloads,
    )

    with patch("rag.migrate.encode_sparse", return_value=MagicMock()):
        result = await run_migration(
            "tenant", "ns",
            "text-embedding-3-small", "text-embedding-3-large",
            _qdrant=qdrant,
            _openai=_make_openai(),
        )

    # Should create final source collection and copy from temp
    assert qdrant.create_collection.call_count == 1
    # Should delete only the temp (source was already gone)
    assert qdrant.delete_collection.call_count == 1
    assert result == 1
