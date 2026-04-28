"""
rag/chunkers/base.py — Abstract base class for all chunkers.

Every chunker receives plain text and returns a list of Chunk objects
ready for embedding. All chunkers set:
  - chunk_id      (UUID v4)
  - content       (≤ 4000 chars)
  - source_id, namespace_id
  - page_number   (if known — PDF only)
  - metadata      {"language": "<iso-code>"}
  - score         0.0 (set at query time, not ingest)
"""

import uuid
from abc import ABC, abstractmethod

from rag.models import Chunk


class BaseChunker(ABC):
    """Common interface for all chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_size: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        """Split text into Chunk objects."""
        ...

    def _make_chunk(
        self,
        content: str,
        source_id: str,
        namespace_id: str,
        language: str,
        page_number: int | None = None,
    ) -> Chunk:
        """Create a single Chunk with all required fields populated."""
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            content=content[:4000],
            source_id=source_id,
            namespace_id=namespace_id,
            score=0.0,
            page_number=page_number,
            metadata={"language": language},
        )
