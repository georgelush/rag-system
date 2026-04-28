"""
rag/migrate.py — Embedding model migration CLI.

Migrates all vectors in a (tenant, namespace) collection from one OpenAI
embedding model to another.

Usage
-----
    python rag/migrate.py \
        --tenant my-tenant \
        --namespace my-docs \
        --from text-embedding-3-small \
        --to text-embedding-3-large

    # Dry-run (no changes):
    python rag/migrate.py --tenant acme --namespace docs \\
        --from text-embedding-3-small --to text-embedding-3-large --dry-run

Migration procedure
-------------------
1.  Scroll all chunk payloads from the source collection (no vectors needed).
2.  Create a temporary collection  <source>__migration_tmp  with the target
    model dimensions.
3.  Re-embed every chunk with the target model (batches of BATCH_SIZE) and
    upsert to the temp collection.
4.  Verify that the temp point count matches the source.
5.  Delete the source collection.
6.  Re-create the source collection with the target dimensions.
7.  Copy all vectors from temp → source (scroll-with-vectors; no re-embedding).
8.  Delete the temp collection.

Resumability
------------
If the process is interrupted between steps 5 and 8 the temp collection will
still exist but the source collection will be gone.  Re-running the script
detects this state and resumes from step 6.
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from rag.sparse_encoder import encode as encode_sparse

load_dotenv()

log = logging.getLogger("rag.migrate")

_SUPPORTED_MODELS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_PROXY: str   = os.environ.get("LLM_PROXY", "")
BATCH_SIZE: int  = 100


# ── Low-level helpers ─────────────────────────────────────────────────────────

async def _scroll_payloads(qdrant, col: str) -> list[dict]:
    """Scroll all points; return [{id, payload}] — no vectors loaded."""
    records: list[dict] = []
    offset = None
    while True:
        result, next_offset = await qdrant.scroll(
            collection_name=col,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in result:
            records.append({"id": str(pt.id), "payload": pt.payload or {}})
        if next_offset is None:
            break
        offset = next_offset
    return records


async def _scroll_with_vectors(qdrant, col: str) -> list:
    """Scroll all points including vectors (for the final copy step)."""
    records = []
    offset = None
    while True:
        result, next_offset = await qdrant.scroll(
            collection_name=col,
            limit=50,           # smaller limit — each record carries two vector arrays
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        records.extend(result)
        if next_offset is None:
            break
        offset = next_offset
    return records


async def _embed_batch(
    openai_client, model: str, texts: list[str]
) -> list[list[float]]:
    resp = await openai_client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


async def _create_collection(qdrant, name: str, dim: int) -> None:
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams
    await qdrant.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    log.info("Created collection '%s' (dim=%d).", name, dim)


# ── Main migration logic ──────────────────────────────────────────────────────

async def run_migration(
    tenant_id: str,
    namespace_id: str,
    from_model: str,
    to_model: str,
    *,
    _qdrant=None,   # injectable for testing
    _openai=None,   # injectable for testing
) -> int:
    """Execute the migration.  Returns the number of chunks migrated."""
    if from_model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported source model {from_model!r}. "
            f"Supported: {list(_SUPPORTED_MODELS)}"
        )
    if to_model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported target model {to_model!r}. "
            f"Supported: {list(_SUPPORTED_MODELS)}"
        )
    if from_model == to_model:
        log.info("Source and target models are identical — nothing to do.")
        return 0

    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import PointStruct
    from openai import AsyncOpenAI

    target_dim = _SUPPORTED_MODELS[to_model]
    source_col = f"{tenant_id}__{namespace_id}"
    temp_col   = f"{source_col}__migration_tmp"

    qdrant = _qdrant
    if qdrant is None:
        openai_kwargs: dict = {"api_key": LLM_API_KEY}
        if LLM_PROXY:
            openai_kwargs["base_url"] = LLM_PROXY
        openai_client = _openai or AsyncOpenAI(**openai_kwargs)
        qdrant = AsyncQdrantClient(url=QDRANT_URL)
    else:
        openai_kwargs = {"api_key": LLM_API_KEY}
        if LLM_PROXY:
            openai_kwargs["base_url"] = LLM_PROXY
        openai_client = _openai or AsyncOpenAI(**openai_kwargs)

    close_qdrant = _qdrant is None  # only close if we created it

    try:
        existing = {c.name for c in (await qdrant.get_collections()).collections}
        source_exists = source_col in existing
        temp_exists   = temp_col in existing

        # Detect interrupted migration: temp present, source gone → resume copy
        resuming = temp_exists and not source_exists
        if resuming:
            log.info(
                "Resuming interrupted migration: temp '%s' exists, source '%s' gone.",
                temp_col, source_col,
            )
        elif not source_exists:
            raise RuntimeError(
                f"Source collection '{source_col}' not found in Qdrant."
            )

        n_chunks = 0

        if not resuming:
            # ── 1. Scroll all chunk payloads ───────────────────────────────────
            log.info("Scrolling chunk payloads from '%s'...", source_col)
            records = await _scroll_payloads(qdrant, source_col)
            n_chunks = len(records)
            log.info("Found %d chunks to migrate.", n_chunks)

            if n_chunks == 0:
                log.info("Collection is empty — nothing to migrate.")
                return 0

            # ── 2. (Re-)create temp collection ─────────────────────────────────
            if temp_exists:
                log.info("Removing stale temp collection '%s'...", temp_col)
                await qdrant.delete_collection(temp_col)
            await _create_collection(qdrant, temp_col, target_dim)

            # ── 3. Re-embed + upsert to temp in batches ────────────────────────
            n_batches = (n_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            for i, start in enumerate(range(0, n_chunks, BATCH_SIZE), 1):
                batch = records[start : start + BATCH_SIZE]
                texts = [r["payload"].get("content", "") for r in batch]
                log.info(
                    "Embedding batch %d/%d (%d chunks)...", i, n_batches, len(batch)
                )
                dense_vecs = await _embed_batch(openai_client, to_model, texts)
                sparse_vecs = [encode_sparse(t) for t in texts]
                points = [
                    PointStruct(
                        id=r["id"],
                        vector={"dense": dv, "sparse": sv},
                        payload=r["payload"],
                    )
                    for r, dv, sv in zip(batch, dense_vecs, sparse_vecs)
                ]
                await qdrant.upsert(collection_name=temp_col, points=points)
                log.info("Upserted %d chunks to temp.", len(points))

            # ── 4. Verify counts ────────────────────────────────────────────────
            src_count  = (await qdrant.get_collection(source_col)).points_count
            temp_count = (await qdrant.get_collection(temp_col)).points_count
            if temp_count != src_count:
                raise RuntimeError(
                    f"Count mismatch after re-embedding: "
                    f"source={src_count}, temp={temp_count}. "
                    "Source collection is unchanged — please retry."
                )
            log.info(
                "Count verified: %d/%d chunks match between source and temp.",
                temp_count, src_count,
            )

            # ── 5. Delete source collection ─────────────────────────────────────
            log.info("Deleting source collection '%s'...", source_col)
            await qdrant.delete_collection(source_col)

        # ── 6. Re-create source with target dimensions ──────────────────────────
        log.info(
            "Re-creating source collection '%s' (dim=%d)...", source_col, target_dim
        )
        await _create_collection(qdrant, source_col, target_dim)

        # ── 7. Copy temp → source (scroll with vectors — no re-embedding) ───────
        log.info("Copying vectors from temp to source...")
        temp_points = await _scroll_with_vectors(qdrant, temp_col)
        n_copied = 0
        for start in range(0, len(temp_points), BATCH_SIZE):
            batch_pts = temp_points[start : start + BATCH_SIZE]
            points = [
                PointStruct(id=str(pt.id), vector=pt.vector, payload=pt.payload)
                for pt in batch_pts
            ]
            await qdrant.upsert(collection_name=source_col, points=points)
            n_copied += len(points)
        log.info("Copied %d chunks to source.", n_copied)

        # ── 8. Delete temp collection ───────────────────────────────────────────
        log.info("Removing temp collection '%s'...", temp_col)
        await qdrant.delete_collection(temp_col)

        log.info(
            "Migration complete: %d chunks migrated from '%s' to '%s' in '%s'.",
            n_copied, from_model, to_model, source_col,
        )
        return n_copied

    finally:
        if close_qdrant:
            await qdrant.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migrate a RAG namespace to a different embedding model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rag/migrate.py \\\n"
            "    --tenant acme --namespace contracts \\\n"
            "    --from text-embedding-3-small --to text-embedding-3-large\n"
        ),
    )
    p.add_argument("--tenant",    required=True, metavar="ID", help="Tenant ID")
    p.add_argument("--namespace", required=True, metavar="ID", help="Namespace ID")
    p.add_argument(
        "--from", dest="from_model", required=True,
        choices=list(_SUPPORTED_MODELS), help="Current embedding model",
    )
    p.add_argument(
        "--to", dest="to_model", required=True,
        choices=list(_SUPPORTED_MODELS), help="Target embedding model",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print chunk count and target dimensions, then exit without changes",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    if args.dry_run:
        async def _dry() -> None:
            from qdrant_client import AsyncQdrantClient
            qdrant = AsyncQdrantClient(url=QDRANT_URL)
            col = f"{args.tenant}__{args.namespace}"
            existing = {
                c.name for c in (await qdrant.get_collections()).collections
            }
            if col not in existing:
                print(f"ERROR: Collection '{col}' not found.")
                await qdrant.close()
                sys.exit(1)
            info = await qdrant.get_collection(col)
            print(f"Dry run for collection '{col}':")
            print(f"  chunks : {info.points_count}")
            print(f"  from   : {args.from_model}  "
                  f"({_SUPPORTED_MODELS[args.from_model]} dims)")
            print(f"  to     : {args.to_model}  "
                  f"({_SUPPORTED_MODELS[args.to_model]} dims)")
            await qdrant.close()

        asyncio.run(_dry())
        sys.exit(0)

    asyncio.run(
        run_migration(
            tenant_id=args.tenant,
            namespace_id=args.namespace,
            from_model=args.from_model,
            to_model=args.to_model,
        )
    )
