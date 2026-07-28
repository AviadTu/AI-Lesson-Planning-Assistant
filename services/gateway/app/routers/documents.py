"""
Document routes: /documents (GET, DELETE), /upload (POST), /documents/download.

The Gateway exposes the public document endpoints and proxies every operation
to the RAG service, which owns the files, the metadata registry and ChromaDB.
Response shapes are preserved from the previous project so the frontend needs
no changes beyond the ``s3_key`` -> ``doc_id`` rename.
"""

from __future__ import annotations

import base64

import httpx
from fastapi import APIRouter, File, Header, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.clients import rag_client
from app.config import settings
from app.database import add_pending_source, get_lesson_mode
from app.extract import ExtractError, extract_text
from app.schemas import DeleteDocumentRequest, ExtractRequest

router = APIRouter()

_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


def _source_kind(filename: str, content_type: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _IMAGE_EXTS or (content_type or "").startswith("image/"):
        return "image"
    return "document"


@router.get("/documents")
async def list_documents():
    try:
        return await rag_client.list_documents()
    except httpx.HTTPError:
        return {"documents": []}


@router.post("/upload")
async def upload(
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    x_session_id: str | None = Header(default=None),
):
    # Accept both the "files" (multiple) and "file" (single) form fields.
    incoming = list(files) if files else []
    if not incoming and file is not None:
        incoming = [file]
    if not incoming:
        return JSONResponse(
            {"error": "No files provided. Use the 'files' field."}, status_code=400
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    read: list[tuple[str, bytes, str]] = []
    for upload_file in incoming:
        content = await upload_file.read()
        if len(content) > max_bytes:
            return JSONResponse(
                {
                    "error": f"File too large. Maximum upload size is "
                    f"{settings.MAX_UPLOAD_MB} MB."
                },
                status_code=413,
            )
        read.append(
            (
                upload_file.filename or "(unnamed)",
                content,
                upload_file.content_type or "application/octet-stream",
            )
        )

    # Route by intent: while a session is in lesson-planning mode, uploads are
    # temporary LESSON SOURCES (held for the next turn -> n8n); otherwise they
    # are permanent KB ingestion via the independent RAG service (unchanged).
    session_id = (x_session_id or "").strip()
    if session_id and get_lesson_mode(session_id).get("active"):
        uploaded = []
        for filename, content, content_type in read:
            add_pending_source(
                session_id,
                _source_kind(filename, content_type),
                filename,
                content_type,
                base64.b64encode(content).decode("ascii"),
            )
            uploaded.append({"original_filename": filename, "doc_id": filename})
        return JSONResponse(
            {"uploaded": uploaded, "errors": [], "ingesting": False,
             "target": "lesson_source"},
            status_code=200,
        )

    try:
        result = await rag_client.upload_documents(read)
    except httpx.HTTPError:
        return JSONResponse({"error": "Upload failed."}, status_code=502)

    status_code = 200 if result.get("uploaded") else 400
    return JSONResponse(result, status_code=status_code)


@router.post("/internal/extract")
async def internal_extract(payload: ExtractRequest):
    """Extract text from a base64 document (used by n8n's DOCX source branch)."""
    try:
        data = base64.b64decode(payload.content_base64)
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid base64 content."}, status_code=400)
    try:
        text = extract_text(payload.filename or "", payload.mime or "", data)
    except ExtractError as exc:
        return JSONResponse({"error": str(exc), "text": ""}, status_code=422)
    return {"text": text, "filename": payload.filename, "mime": payload.mime}


@router.delete("/documents")
async def delete_document(payload: DeleteDocumentRequest):
    doc_id = (payload.doc_id or "").strip()
    if not doc_id:
        return JSONResponse({"error": "doc_id is required."}, status_code=400)
    try:
        result = await rag_client.delete_document(doc_id)
    except httpx.HTTPError:
        return JSONResponse({"error": "Failed to delete document."}, status_code=502)
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)


@router.get("/documents/download")
async def download_document(doc_id: str = ""):
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return JSONResponse({"error": "doc_id is required."}, status_code=400)

    client, request = rag_client.download_stream(doc_id)
    upstream = await client.send(request, stream=True)

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return JSONResponse(
            {"error": "Could not download document."},
            status_code=upstream.status_code,
        )

    async def _iter():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    # Preserve the Content-Disposition (RFC 5987 UTF-8 filename) from the RAG
    # service so Hebrew filenames download correctly.
    headers = {}
    disposition = upstream.headers.get("content-disposition")
    if disposition:
        headers["Content-Disposition"] = disposition
    media_type = upstream.headers.get("content-type", "application/octet-stream")

    return StreamingResponse(_iter(), media_type=media_type, headers=headers)
