"""Test ingest of hr_handbook.txt directly against the RAG service."""
import json
import time
import uuid
import httpx

RAG_URL    = "http://localhost:8080"
API_KEY    = "rag-system-dev-key-change-in-production"
TENANT_ID  = "acme"
NAMESPACE  = "hr-handbook"
FILE_PATH  = r"C:\Users\a866079\OneDrive - ATOS\Documents\SpectrumAI\spectrumai-agent-framework\agents\implementations\hr_assistant\data\hr_handbook.txt"

idempotency_key = str(uuid.uuid4())
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Request-ID": idempotency_key,
    "X-Tenant-ID": TENANT_ID,
    "Idempotency-Key": idempotency_key,
}
payload = {
    "namespace_id": NAMESPACE,
    "source_id": "hr_handbook",
    "source_type": "file",
}

print("=== Submitting ingest job ===")
with open(FILE_PATH, "rb") as fh:
    file_bytes = fh.read()
print(f"File size: {len(file_bytes):,} bytes")

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{RAG_URL}/v1/ingest",
        headers=headers,
        data={"payload": json.dumps(payload)},
        files={"file": ("hr_handbook.txt", file_bytes)},
    )
print(f"Submit status: {resp.status_code}")
if resp.status_code >= 400:
    print(f"Error: {resp.text}")
    exit(1)

job_id = resp.json()["job_id"]
print(f"Job ID: {job_id}")

print("\n=== Polling for completion (max 90s) ===")
deadline = time.time() + 90
poll_headers = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Request-ID": str(uuid.uuid4()),
    "X-Tenant-ID": TENANT_ID,
}
while time.time() < deadline:
    time.sleep(3)
    with httpx.Client(timeout=10.0) as client:
        poll = client.get(f"{RAG_URL}/v1/ingest/{job_id}", headers=poll_headers)
    data = poll.json()
    status = data.get("status", "")
    stage  = data.get("progress", {}).get("stage", "")
    pct    = data.get("progress", {}).get("percent", 0)
    print(f"  [{status}] stage={stage} {pct}%")
    if status == "done":
        chunks = data.get("progress", {}).get("chunks_created", "?")
        print(f"\n✓ SUCCESS — {chunks} chunks indexed into {TENANT_ID}/{NAMESPACE}")
        break
    if status == "failed":
        print(f"\n✗ FAILED — {json.dumps(data.get('error'))}")
        break
else:
    print(f"\n⚠ TIMEOUT after 90s (job: {job_id})")

print("\n=== Qdrant collection check ===")
import urllib.request
req = urllib.request.urlopen("http://localhost:6333/collections/acme__hr-handbook", timeout=5)
col = json.loads(req.read())
print(f"  points_count: {col['result']['points_count']}")
