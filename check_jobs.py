import asyncio, json
import redis.asyncio as aioredis

async def check():
    r = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
    keys = await r.keys("rag:job:*")
    for k in sorted(keys)[-5:]:
        val = await r.get(k)
        if val:
            j = json.loads(val)
            status = j.get("status", "")
            error = j.get("error", None)
            print(k, "|", status, "|", json.dumps(error))
    await r.aclose()

asyncio.run(check())
