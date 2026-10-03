import os
import uuid

import psycopg
import pytest

os.environ.setdefault("EMBEDDER", "fake")

from app.config import settings  # noqa: E402
from app.ingestion.embeddings import FakeEmbedder, to_pgvector  # noqa: E402


def _dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://").replace("postgresql+asyncpg://", "postgresql://")


OWNER_DSN = _dsn(settings.owner_database_url)
APP_DSN = _dsn(settings.app_database_url)
WORKER_DSN = _dsn(settings.worker_database_url)


def _db_available() -> bool:
    try:
        with psycopg.connect(OWNER_DSN, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


requires_db = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable (run `make up`)")


@pytest.fixture
def world():
    """Two fresh tenants with overlapping content. Inserted as OWNER; removed afterwards."""
    emb = FakeEmbedder(settings.embedding_dim)
    w = {"tenants": {}, "docs": {}}
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        for name in ("a", "b"):
            slug = f"t-{name}-{uuid.uuid4().hex[:8]}"
            tid = conn.execute("INSERT INTO tenants (slug, name) VALUES (%s, %s) RETURNING id", (slug, slug)).fetchone()[0]
            w["tenants"][name] = str(tid)
            for key, acl, content in (
                ("salary", ["role:hr"], f"tenant {name} salary bands merit pool"),
                ("runbook", ["group:engineering"], f"tenant {name} on-call runbook latency"),
                ("handbook", ["role:employee"], f"tenant {name} handbook leave policy"),
            ):
                did = conn.execute(
                    "INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by) "
                    "VALUES (%s, %s, 'text/plain', %s, 'x', %s, gen_random_uuid()) RETURNING id",
                    (tid, key, content.encode(), acl),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (%s, 0, %s, %s::vector)",
                    (did, content, to_pgvector(emb.embed_query(content))),
                )
                w["docs"][(name, key)] = str(did)
    yield w
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM tenants WHERE id = ANY(%s::uuid[])", (list(w["tenants"].values()),))


def as_role(dsn: str, tenant_id: str | None, principals: list[str] | None = None, user_id: str | None = None):
    """Open a connection with a transaction whose RLS context is set like the app sets it."""
    conn = psycopg.connect(dsn)
    if tenant_id is not None:
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
    if principals is not None:
        conn.execute("SELECT set_config('app.principals', %s, true)", (",".join(principals),))
    if user_id is not None:
        conn.execute("SELECT set_config('app.user_id', %s, true)", (user_id,))
    return conn
