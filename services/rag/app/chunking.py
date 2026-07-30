"""
Chunking for extracted document segments.

A character-based sliding window with configurable size and overlap. The window
end is snapped to a natural boundary (paragraph → line → sentence → space) when
possible so words/sentences are not cut mid-token; this works for Hebrew because
splitting is purely on Unicode code points and separators, never on encoding.

Each produced chunk carries a document-global ``chunk_index`` (0-based) and the
``page_number`` of the segment it came from (``None`` when unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.extraction import Segment

# Boundaries preferred when snapping a window end, from strongest to weakest.
_BOUNDARIES = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int | None


def chunk_document(
    segments: list[Segment],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split every segment into overlapping chunks with a global chunk index."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    chunks: list[Chunk] = []
    index = 0
    for segment in segments:
        for piece in _split_text(segment.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, chunk_index=index, page_number=segment.page_number)
            )
            index += 1
    return chunks


def chunk_document_single(segments: list[Segment]) -> list[Chunk]:
    """
    Presentation mode: represent the WHOLE document as exactly one chunk.

    All segments (e.g. PDF pages) are joined in order into a single text so the
    embedding represents the complete document and headings/sections/activities
    are never split into separate chunks. Assumes short documents (<= ~3 pages).
    """
    text = "\n\n".join(
        (s.text or "").strip() for s in segments if (s.text or "").strip()
    ).strip()
    if not text:
        return []
    return [Chunk(text=text, chunk_index=0, page_number=None)]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window split with boundary snapping and overlap."""
    text = (text or "").strip()
    n = len(text)
    if n == 0:
        return []
    if n <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in _BOUNDARIES:
                pos = window.rfind(sep)
                # Only snap if the boundary is past the halfway point so we do
                # not create tiny chunks.
                if pos != -1 and pos >= size // 2:
                    end = start + pos + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= n:
            break
        # Step forward, retaining `overlap` characters of context.
        start = max(end - overlap, start + 1)

    return pieces
