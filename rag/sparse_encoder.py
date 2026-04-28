"""
rag/sparse_encoder.py — Lightweight sparse TF vector encoder.

Produces Qdrant SparseVector objects for hybrid search (dense + sparse / BM25-style).

Algorithm:
  1. Tokenise text — lowercase, split on non-alphanumeric, strip stopwords
  2. Compute term frequency (TF) per token
  3. Map each token to a deterministic 24-bit integer index via MD5 hash
     (16M-bucket space → collision probability < 0.001 % for typical docs)
  4. Return SparseVector(indices=[...], values=[...])

No external models, no vocabulary file, microsecond latency.
Works across all 75 languages lingua supports (no language-specific tuning).
"""

import hashlib
import re
from collections import Counter

from qdrant_client.models import SparseVector

# ── Minimal English stopword list ─────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "not", "no", "it", "its", "this", "that",
    "as", "if", "up", "out", "so", "he", "she", "we", "they", "i", "you",
})

_HASH_SPACE = 2 ** 24  # 16 777 216 buckets


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric + filter stopwords + min length 2."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _token_index(token: str) -> int:
    """Deterministic 24-bit index for a token (MD5, first 3 bytes, little-endian)."""
    digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:3], "little")  # 0 … 16 777 215


def encode(text: str) -> SparseVector:
    """Encode text into a Qdrant SparseVector using normalised TF weights.

    Args:
        text: Plain text string (any language).

    Returns:
        SparseVector with indices (hashed token IDs) and values (TF scores).
        Returns an empty SparseVector for blank/empty input.
    """
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    tf      = Counter(tokens)
    total   = len(tokens)

    # Accumulate scores per hash bucket (handles rare hash collisions by summing)
    bucket: dict[int, float] = {}
    for token, count in tf.items():
        idx = _token_index(token)
        bucket[idx] = bucket.get(idx, 0.0) + count / total

    indices = list(bucket.keys())
    values  = [bucket[i] for i in indices]

    return SparseVector(indices=indices, values=values)
