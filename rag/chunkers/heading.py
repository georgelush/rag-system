"""
rag/chunkers/heading.py — Section/heading-aware chunker.

Detects numbered sections, Markdown headings, and structural labels.
Language-agnostic — works on any document with a recognisable outline.

Detected patterns (case-insensitive, must appear at line start):
  - Markdown headings:   # Title / ## Sub / ### Sub-sub
  - Numbered sections:   1.  /  1.1.  /  2.3.1.
  - Section signs:       § 5  /  §5
  - English labels:      Chapter 3, Section 4, Part II, Appendix A, Annex B
  - Upper-case variants: CHAPTER 1, SECTION 2, PART III
"""

import re

from rag.chunkers.base import BaseChunker
from rag.models import Chunk

_HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+"                                                # Markdown # Title
    r"|(?:\d+\.){1,4}\s+"                                       # 1. / 1.1. / 2.3.1.
    r"|§\s*\d+"                                                 # § 5 / §5
    r"|(?:Chapter|Section|Part|Appendix|Annex)\s+\w+"          # English labels
    r"|(?:CHAPTER|SECTION|PART|APPENDIX|ANNEX)\s+\w+"          # Upper-case
    r")",
)


class HeadingChunker(BaseChunker):
    """One chunk per section. Oversized sections are subdivided via ParagraphChunker."""

    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        splits = [m.start() for m in _HEADING_RE.finditer(text)]

        if len(splits) < 2:
            # Not enough headings found — fall back to paragraph
            from rag.chunkers.paragraph import ParagraphChunker
            return ParagraphChunker(self.chunk_size, self.overlap, self.min_chunk_size).chunk(
                text, source_id, namespace_id, language
            )

        boundaries = [0] + splits + [len(text)]
        chunks: list[Chunk] = []

        for i in range(len(boundaries) - 1):
            section = text[boundaries[i]:boundaries[i + 1]].strip()
            if not section or len(section) < self.min_chunk_size:
                continue

            if len(section) <= self.chunk_size:
                chunks.append(self._make_chunk(section, source_id, namespace_id, language))
            else:
                # Section too large — subdivide with paragraph
                from rag.chunkers.paragraph import ParagraphChunker
                sub = ParagraphChunker(self.chunk_size, self.overlap, self.min_chunk_size)
                chunks.extend(sub.chunk(section, source_id, namespace_id, language))

        return chunks
