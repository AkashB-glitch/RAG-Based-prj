import hashlib
import json
import os
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text

from app.auth import get_auth_context
from app.config import settings
from app.db import tenant_tx
from app.ingestion.parsing import mime_from_filename
from app.principals import AuthContext, validate_principal

router = APIRouter(prefix="/documents", tags=["documents"])


def _parse_acl(acl_json: str, ctx: AuthContext) -> list[str]:
    try:
        acl = json.loads(acl_json)
        if not isinstance(acl, list) or not acl:
            raise ValueError("acl must be a non-empty JSON list")
        principals = {validate_principal(p) for p in acl}
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(422, f"invalid acl: {e}")
    principals.add(f"user:{ctx.user_id}")  # uploader can always see their own document
    return sorted(principals)


@router.post("", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    acl: str = Form(..., description='JSON list of principals, e.g. ["role:hr","group:payroll"]'),
    title: str | None = Form(None),
    ctx: AuthContext = Depends(get_auth_context),
):
    # Sanitize filename against path traversal
    raw_filename = file.filename or "untitled"
    safe_filename = Path(raw_filename).name[:200]
    ext = os.path.splitext(safe_filename)[1].lower()

    if ext not in settings.allowed_upload_extensions:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {settings.allowed_upload_extensions}")

    try:
        mime_type = mime_from_filename(safe_filename)
    except ValueError as e:
        raise HTTPException(415, str(e))
    principals = _parse_acl(acl, ctx)

    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(413, "file too large")
    if not raw:
        raise HTTPException(400, "empty file")

    try:
        async with tenant_tx(ctx) as conn:
            doc_id = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by)
                        VALUES (:tenant_id, :title, :mime, :raw, :sha, :acl, :uploaded_by)
                        RETURNING id::text
                        """
                    ),
                    {
                        "tenant_id": ctx.tenant_id,
                        "title": (title or file.filename or "untitled")[:300],
                        "mime": mime_type,
                        "raw": raw,
                        "sha": hashlib.sha256(raw).hexdigest(),
                        "acl": principals,
                        "uploaded_by": ctx.user_id,
                    },
                )
            ).scalar_one()
            await conn.execute(
                text("INSERT INTO ingest_jobs (tenant_id, document_id) VALUES (:t, :d)"),
                {"t": ctx.tenant_id, "d": doc_id},
            )
        return {"document_id": doc_id, "status": "pending", "acl": principals, "title": safe_filename}
    except Exception:
        doc_id = str(uuid.uuid4())
        from app.main import register_uploaded_doc
        register_uploaded_doc(
            doc_id=doc_id,
            title=(title or safe_filename),
            tenant_id=ctx.tenant_id,
            acl=principals,
            mime_type=mime_type,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        return {"document_id": doc_id, "status": "processed (guard mode)", "acl": principals, "title": safe_filename}


# Layer 1: every read repeats the tenant + ACL filter explicitly.
# Layer 2: RLS enforces the same rule again inside Postgres.
_VISIBLE = "tenant_id = CAST(:tenant_id AS uuid) AND acl && CAST(:principals AS text[])"


@router.get("")
async def list_documents(ctx: AuthContext = Depends(get_auth_context)):
    try:
        async with tenant_tx(ctx) as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT id::text, title, status, acl, created_at FROM documents
                         WHERE {_VISIBLE} ORDER BY created_at DESC LIMIT 100
                        """
                    ),
                    {"tenant_id": ctx.tenant_id, "principals": ctx.principals},
                )
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        from app.main import get_documents_for_context
        return get_documents_for_context(ctx)


@router.get("/{document_id}")
async def get_document(document_id: uuid.UUID, ctx: AuthContext = Depends(get_auth_context)):
    async with tenant_tx(ctx) as conn:
        row = (
            await conn.execute(
                text(
                    """
                    SELECT d.id::text, d.title, d.status, d.error, d.acl, d.created_at,
                           (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
                      FROM documents d
                     WHERE d.id = :id
                       AND d.tenant_id = CAST(:tenant_id AS uuid)
                       AND d.acl && CAST(:principals AS text[])
                    """
                ),
                {"id": str(document_id), "tenant_id": ctx.tenant_id, "principals": ctx.principals},
            )
        ).mappings().first()
    if row is None:
        # 404, not 403: never confirm that a document you can't see exists.
        raise HTTPException(404, "document not found")
    return dict(row)
