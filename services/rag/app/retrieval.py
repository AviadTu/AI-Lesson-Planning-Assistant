"""
Retrieval layer.

Embeds the user query with the existing Ollama embedding client, runs a
nearest-neighbour search against ChromaDB (native cosine distance), converts
distance to a cosine similarity (``similarity = 1 - distance``) and drops any
chunk below the configured similarity threshold.

No LLM is involved: this module only finds and returns relevant chunks.
"""

from __future__ import annotations

from app.config import settings
from app.embeddings import ollama_embeddings
from app.vectorstore import chroma


class EmptyQueryError(ValueError):
    """Raised when the query is missing/blank."""


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the query embedding cannot be produced (e.g. Ollama down)."""


def retrieve(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """
    Return the most relevant chunks for ``query``.

    Each returned dict contains: text, doc_id, original_filename, chunk_index,
    page_number (when available) and similarity. Results are ordered by
    descending similarity and exclude anything below the threshold.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        raise EmptyQueryError("Query must not be empty.")

    k = settings.RETRIEVAL_K if top_k is None else int(top_k)
    if k < 1:
        raise ValueError("top_k must be >= 1.")

    thresh = settings.SIMILARITY_THRESHOLD if threshold is None else float(threshold)

    try:
        query_embedding = ollama_embeddings.embed_query(cleaned)
    except Exception as exc:  # noqa: BLE001 — surface as a clear service error
        raise EmbeddingUnavailableError(str(exc)) from exc
    rows = chroma.query_chunks(query_embedding, k)

    results: list[dict] = []
    for row in rows:
        similarity = 1.0 - float(row["distance"])
        if similarity < thresh:
            continue
        results.append(
            {
                "text": row["text"],
                "doc_id": row["doc_id"],
                "original_filename": row["original_filename"],
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "similarity": round(similarity, 6),
            }
        )
    return results
