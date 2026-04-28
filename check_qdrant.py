import asyncio
from qdrant_client import AsyncQdrantClient

async def check():
    client = AsyncQdrantClient(url="http://localhost:6333")
    info = await client.get_collection("acme__hr-handbook")
    params = info.config.params
    print("params type:", type(params))
    print("params attrs:", [a for a in dir(params) if not a.startswith("_")])
    # Try both possible attribute names
    for attr in ("vectors_config", "vectors", "sparse_vectors_config", "sparse_vectors"):
        val = getattr(params, attr, "MISSING")
        print(f"  {attr}: {type(val).__name__} = {repr(val)[:80]}")

asyncio.run(check())
