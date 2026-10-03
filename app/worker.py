"""Ingestion worker: claim job -> parse -> chunk -> embed -> write chunks.

Runs as the `rag_worker` DB role. RLS lets it touch only rows of the tenant it has set in
app.tenant_id, and a DB trigger copies tenant_id + ACL from the parent document onto every
chunk, so the worker cannot write a chunk with the wrong permissions even if it tried.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.ingestion.chunking import chunk_text
from app.ingestion.embeddings import get_embedder, to_pgvector
from app.ingestion.parsing import extract_text

log = logging.getLogger("worker")
engine = create_async_engine(settings.worker_database_url, pool_pre_ping=True, pool_size=4)

CLAIM_SQL = text(
    """
    UPDATE ingest_jobs
       SET status = 'running', attempts = attempts + 1, locked_at = now()
     WHERE id = (
        SELECT id FROM ingest_jobs
         WHERE (status = 'queued' AND run_after <= now())
            OR (status = 'running' AND locked_at < now() - make_interval(mins => :stale))
         ORDER BY id
         FOR UPDATE SKIP LOCKED
         LIMIT 1)
    RETURNING id, tenant_id::text AS tenant_id, document_id::text AS document_id, attempts
    """
)


@asynccontextmanager
async def worker_tx(tenant_id: str):
    async with engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_id})
        yield conn


async def claim_job():
    async with engine.begin() as conn:
        return (await conn.execute(CLAIM_SQL, {"stale": settings.worker_stale_lock_minutes})).mappings().first()


async def process(job) -> int:
    tenant_id, doc_id = job["tenant_id"], job["document_id"]

    async with worker_tx(tenant_id) as conn:
        doc = (
            await conn.execute(
                text("SELECT raw, mime_type FROM documents WHERE id = :id"), {"id": doc_id}
            )
        ).mappings().first()
        if doc is None:
            raise LookupError("document not visible to worker (deleted, or tenant mismatch)")
        await conn.execute(
            text("UPDATE documents SET status='processing', updated_at=now() WHERE id=:id"), {"id": doc_id}
        )

    body = extract_text(bytes(doc["raw"]), doc["mime_type"])
    pieces = chunk_text(body, settings.chunk_max_chars, settings.chunk_overlap_chars)
    if not pieces:
        raise ValueError("no extractable text")
    vectors = await asyncio.to_thread(get_embedder().embed_documents, pieces)

    async with worker_tx(tenant_id) as conn:
        await conn.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": doc_id})  # idempotent retries
        await conn.execute(
            text(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding)
                VALUES (:document_id, :chunk_index, :content, CAST(:embedding AS vector))
                """  # tenant_id and acl are filled by trigger from the parent document
            ),
            [
                {"document_id": doc_id, "chunk_index": i, "content": c, "embedding": to_pgvector(v)}
                for i, (c, v) in enumerate(zip(pieces, vectors))
            ],
        )
        await conn.execute(
            text("UPDATE documents SET status='ready', error=NULL, updated_at=now() WHERE id=:id"), {"id": doc_id}
        )
        await conn.execute(
            text("UPDATE ingest_jobs SET status='done', locked_at=NULL, last_error=NULL WHERE id=:id"),
            {"id": job["id"]},
        )
    return len(pieces)


async def fail(job, exc: Exception) -> None:
    err = f"{type(exc).__name__}: {exc}"[:1000]
    final = job["attempts"] >= settings.worker_max_attempts
    async with worker_tx(job["tenant_id"]) as conn:
        await conn.execute(
            text(
                """
                UPDATE ingest_jobs
                   SET status = :status, locked_at = NULL, last_error = :err,
                       run_after = now() + make_interval(secs => :backoff)
                 WHERE id = :id
                """
            ),
            {
                "status": "failed" if final else "queued",
                "err": err,
                "backoff": min(2 ** job["attempts"], 300),
                "id": job["id"],
            },
        )
        await conn.execute(
            text("UPDATE documents SET status=:s, error=:err, updated_at=now() WHERE id=:id"),
            {"s": "failed" if final else "pending", "err": err, "id": job["document_id"]},
        )


async def run_once() -> bool:
    """Process at most one job. Returns True if a job was claimed."""
    job = await claim_job()
    if job is None:
        return False
    try:
        n = await process(job)
        log.info("job=%s doc=%s ingested chunks=%d", job["id"], job["document_id"], n)
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded on the job
        log.exception("job=%s failed (attempt %d)", job["id"], job["attempts"])
        await fail(job, exc)
    return True


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await asyncio.to_thread(get_embedder)  # load/download the model before claiming work
    log.info("worker started (embedder=%s)", settings.embedder)
    while True:
        if not await run_once():
            await asyncio.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
