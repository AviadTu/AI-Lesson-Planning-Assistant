"""
Document routes owned by the RAG service:
  POST   /documents/upload   — save file(s) + register + index into ChromaDB
  GET    /documents          — list documents (registry, newest first)
  DELETE /documents          — delete file + registry row + Chroma vectors
  GET    /documents/download — stream a document back (forces download)

File storage and the metadata registry are fully functional. On upload each
document is ingested into ChromaDB on a background thread (extract → chunk →
embed → store); deletion removes the file, the registry row and all Chroma
chunks.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.ingestion import ingest_document_async
from app.schemas import DeleteDocumentRequest
from app.storage import files as file_store
from app.storage import registry
from app.vectorstore import chroma

router = APIRouter()
logger = logging.getLogger("rag.documents")


@router.post("/documents/upload")
async def upload(files: list[UploadFile] = File(default=[])):
    if not files:
        return JSONResponse(
            {"error": "No files provided. Use the 'files' field."}, status_code=400
        )

    uploaded: list[dict] = []
    errors: list[dict] = []
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    for upload_file in files:
        filename = upload_file.filename or "(unnamed)"
        try:
            content = await upload_file.read()
            if len(content) > max_bytes:
                errors.append(
                    {
                        "filename": filename,
                        "error": f"File too large. Maximum upload size is "
                        f"{settings.MAX_UPLOAD_MB} MB.",
                    }
                )
                continue

            original_filename, doc_id = file_store.save_upload(
                filename, content, upload_file.content_type
            )
            registry.record_document(original_filename, doc_id)
            # Ingest into ChromaDB on a background thread; overwriting a file
            # (same doc_id) replaces its chunks (handled in the vector store).
            ingest_document_async(
                doc_id, original_filename, str(file_store.resolve_path(doc_id))
            )
            uploaded.append({"original_filename": original_filename, "doc_id": doc_id})
        except file_store.UploadError as exc:
            errors.append({"filename": filename, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("upload failed for %s: %s", filename, exc)
            errors.append({"filename": filename, "error": "Upload failed."})

    return {"uploaded": uploaded, "errors": errors, "ingesting": bool(uploaded)}


@router.get("/documents")
async def list_documents():
    return {"documents": registry.list_documents()}


@router.delete("/documents")
def delete_document(payload: DeleteDocumentRequest):
    # Sync route: the Chroma delete call runs in a worker thread, not on the
    # event loop.
    doc_id = (payload.doc_id or "").strip()
    if not doc_id:
        return JSONResponse({"error": "doc_id is required."}, status_code=400)

    try:
        file_store.delete_file(doc_id)
    except file_store.UploadError:
        return JSONResponse({"error": "Invalid doc_id."}, status_code=400)

    registry.delete_document(doc_id)
    # Remove all vectors for this document synchronously (fast, local).
    chroma.delete_document_vectors(doc_id)
    return {"ok": True, "ingesting": False}


@router.get("/documents/download")
async def download_document(doc_id: str = ""):
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return JSONResponse({"error": "doc_id is required."}, status_code=400)

    try:
        path = file_store.resolve_path(doc_id)
    except file_store.UploadError:
        return JSONResponse({"error": "Invalid doc_id."}, status_code=400)

    if not path.exists():
        return JSONResponse({"error": "Document not found."}, status_code=404)

    # Force download and preserve the original (possibly Hebrew) filename via
    # RFC 5987, with an ASCII fallback for old clients.
    display_name = registry.get_display_name(doc_id) or doc_id
    ascii_fallback = display_name.encode("ascii", "ignore").decode("ascii") or "document"
    disposition = (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(display_name, safe='')}"
    )
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": disposition},
    )
