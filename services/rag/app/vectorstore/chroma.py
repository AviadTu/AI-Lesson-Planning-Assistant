"""
ChromaDB vector store (local, persistent).

Stores one record per chunk: the chunk text, its precomputed embedding, and
metadata ``{doc_id, original_filename, chunk_index, page_number?}``. Embeddings
are computed by the Ollama embeddings module and passed in, so Chroma never
needs its own embedding function.

Writes are serialised with a lock so concurrent ingestion threads cannot race
on the same collection.
"""

from __future__ import annotations

import threading

from app.config import settings
from app.chunking import Chunk

_client = None
_collection = None
# Reentrant: get_collection() holds this lock and then calls get_client(),
# which acquires it again on the same thread.
_init_lock = threading.RLock()
_write_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                import chromadb

                _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        with _init_lock:
            if _collection is None:
                # embedding_function=None: we always supply precomputed
                # embeddings (Ollama), so Chroma must NOT instantiate its
                # default ONNX model (which would try to download one).
                _collection = get_client().get_or_create_collection(
                    name=settings.CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
    return _collection


def delete_document_vectors(doc_id: str) -> None:
    """Remove all chunks belonging to a document."""
    with _write_lock:
        get_collection().delete(where={"doc_id": doc_id})


def replace_document(
    doc_id: str,
    original_filename: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """
    Replace a document's vectors: delete existing chunks for ``doc_id`` first
    (overwrite synchronisation), then insert the new chunks. Returns the number
    of chunks stored.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch.")

    with _write_lock:
        collection = get_collection()
        # Always clear old chunks first so re-ingestion never leaves stale data.
        collection.delete(where={"doc_id": doc_id})

        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for chunk in chunks:
            ids.append(f"{doc_id}::{chunk.chunk_index}")
            documents.append(chunk.text)
            meta: dict = {
                "doc_id": doc_id,
                "original_filename": original_filename,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page_number is not None:
                meta["page_number"] = chunk.page_number
            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)


def query_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Nearest-neighbour search against the collection using Chroma's native
    cosine distance (the collection is created with ``hnsw:space=cosine``).

    Returns up to ``top_k`` rows, each a dict with the chunk text, its metadata
    and the raw Chroma ``distance``. Similarity is derived by the caller as
    ``1 - distance``. Reads always reflect the current collection state, so
    deleted/overwritten chunks are never returned.
    """
    collection = get_collection()
    available = collection.count()
    if available == 0:
        return []

    n_results = max(1, min(top_k, available))
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    rows: list[dict] = []
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        rows.append(
            {
                "text": documents[i],
                "doc_id": meta.get("doc_id"),
                "original_filename": meta.get("original_filename"),
                "chunk_index": meta.get("chunk_index"),
                "page_number": meta.get("page_number"),
                "distance": distances[i],
            }
        )
    return rows


def counts() -> dict:
    """Return total chunk count and distinct document count."""
    collection = get_collection()
    chunk_count = collection.count()
    doc_ids: set[str] = set()
    if chunk_count:
        got = collection.get(include=["metadatas"])
        for meta in got.get("metadatas", []) or []:
            if meta and meta.get("doc_id"):
                doc_ids.add(meta["doc_id"])
    return {"document_count": len(doc_ids), "chunk_count": chunk_count}


def document_summaries() -> list[dict]:
    """
    Per-document diagnostic summary: chunk count, page numbers present, and a
    short sample of the first chunk's text (to confirm Hebrew is preserved).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    got = collection.get(include=["metadatas", "documents"])
    metas = got.get("metadatas", []) or []
    docs = got.get("documents", []) or []

    by_doc: dict[str, dict] = {}
    for meta, text in zip(metas, docs):
        if not meta:
            continue
        doc_id = meta.get("doc_id", "")
        entry = by_doc.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "original_filename": meta.get("original_filename"),
                "chunk_count": 0,
                "pages": set(),
                "_first_text": None,
                "_first_index": None,
            },
        )
        entry["chunk_count"] += 1
        if meta.get("page_number") is not None:
            entry["pages"].add(meta["page_number"])
        idx = meta.get("chunk_index")
        if idx is not None and (entry["_first_index"] is None or idx < entry["_first_index"]):
            entry["_first_index"] = idx
            entry["_first_text"] = text

    summaries: list[dict] = []
    for entry in by_doc.values():
        sample = (entry.pop("_first_text") or "")[:80]
        entry.pop("_first_index", None)
        entry["pages"] = sorted(entry["pages"])
        entry["sample_text"] = sample
        summaries.append(entry)
    return summaries
