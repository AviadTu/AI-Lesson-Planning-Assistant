"""
Document routes: /documents (GET, DELETE), /upload (POST), /documents/download.

The Gateway exposes the public document endpoints and proxies every operation
to the RAG service, which owns the files, the metadata registry and ChromaDB.
Response shapes are preserved from the previous project so the frontend needs
no changes beyond the ``s3_key`` -> ``doc_id`` rename.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.clients import rag_client
from app.config import settings
from app.schemas import DeleteDocumentRequest

router = APIRouter()


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
    payload: list[tuple[str, bytes, str]] = []
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
        payload.append(
            (
                upload_file.filename or "(unnamed)",
                content,
                upload_file.content_type or "application/octet-stream",
            )
        )

    try:
        result = await rag_client.upload_documents(payload)
    except httpx.HTTPError:
        return JSONResponse({"error": "Upload failed."}, status_code=502)

    status_code = 200 if result.get("uploaded") else 400
    return JSONResponse(result, status_code=status_code)


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
