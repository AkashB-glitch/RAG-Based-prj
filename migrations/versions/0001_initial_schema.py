"""Initial schema: tenants, users, documents, chunks, ingest_jobs, retrieval_audit + RLS.

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- RLS context helpers
-- Missing context -> NULL tenant / empty principals -> policies match nothing (fail closed).
CREATE FUNCTION app_tenant() RETURNS uuid LANGUAGE sql STABLE AS
$$ SELECT nullif(current_setting('app.tenant_id', true), '')::uuid $$;

CREATE FUNCTION app_user_id() RETURNS text LANGUAGE sql STABLE AS
$$ SELECT nullif(current_setting('app.user_id', true), '') $$;

CREATE FUNCTION app_principals() RETURNS text[] LANGUAGE sql STABLE AS
$$ SELECT coalesce(string_to_array(nullif(current_setting('app.principals', true), ''), ','), '{}'::text[]) $$;

-- ---------------------------------------------------------------- tables
CREATE TABLE tenants (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug       text NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9-]{2,40}$'),
    name       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email      text NOT NULL,
    roles      text[] NOT NULL DEFAULT '{}',
    groups     text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

CREATE TABLE documents (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    title          text NOT NULL,
    mime_type      text NOT NULL,
    raw            bytea NOT NULL,
    content_sha256 text NOT NULL,
    acl            text[] NOT NULL CHECK (cardinality(acl) > 0),
    status         text NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error          text,
    uploaded_by    uuid NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX documents_tenant_idx ON documents (tenant_id);
CREATE INDEX documents_acl_idx    ON documents USING gin (acl);

CREATE TABLE chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,
    acl         text[] NOT NULL,                       -- denormalized from documents (ADR-002)
    content     text NOT NULL,
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_acl_idx       ON chunks USING gin (acl);
CREATE INDEX chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX chunks_tenant_idx    ON chunks (tenant_id);

-- A chunk's tenant and ACL are ALWAYS copied from its parent document. Callers cannot choose them.
-- Runs as the inserting role, so the document lookup is itself subject to RLS: a worker whose
-- app.tenant_id is tenant A cannot attach a chunk to tenant B's document (lookup finds nothing).
CREATE FUNCTION chunks_inherit_acl() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE d record;
BEGIN
    SELECT tenant_id, acl INTO d FROM documents WHERE id = NEW.document_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'document % not found or not visible', NEW.document_id
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    NEW.tenant_id := d.tenant_id;
    NEW.acl := d.acl;
    RETURN NEW;
END $$;
CREATE TRIGGER chunks_inherit_acl BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_inherit_acl();

CREATE TABLE ingest_jobs (          -- holds only ids, no content -> no RLS needed
    id          bigserial PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts    int  NOT NULL DEFAULT 0,
    run_after   timestamptz NOT NULL DEFAULT now(),
    locked_at   timestamptz,
    last_error  text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ingest_jobs_queued_idx ON ingest_jobs (run_after) WHERE status = 'queued';

CREATE TABLE retrieval_audit (      -- append-only; written in the same tx as retrieval (week 2)
    id           bigserial PRIMARY KEY,
    request_id   uuid NOT NULL,
    tenant_id    uuid NOT NULL REFERENCES tenants(id),
    user_id      text NOT NULL,
    principals   text[] NOT NULL,
    query_sha256 text NOT NULL,
    query_text   text,
    chunk_ids    uuid[] NOT NULL,
    document_ids uuid[] NOT NULL,
    latency_ms   int,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX retrieval_audit_tenant_time_idx ON retrieval_audit (tenant_id, created_at DESC);

-- ---------------------------------------------------------------- row-level security
-- FORCE makes policies apply to the table owner too (superusers still bypass: never
-- connect the API as a superuser).
ALTER TABLE users           ENABLE ROW LEVEL SECURITY;  ALTER TABLE users           FORCE ROW LEVEL SECURITY;
ALTER TABLE documents       ENABLE ROW LEVEL SECURITY;  ALTER TABLE documents       FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks          ENABLE ROW LEVEL SECURITY;  ALTER TABLE chunks          FORCE ROW LEVEL SECURITY;
ALTER TABLE retrieval_audit ENABLE ROW LEVEL SECURITY;  ALTER TABLE retrieval_audit FORCE ROW LEVEL SECURITY;

-- API role (app_user): tenant AND ACL.
CREATE POLICY users_app_read ON users FOR SELECT TO app_user
    USING (tenant_id = app_tenant());

CREATE POLICY documents_app_read ON documents FOR SELECT TO app_user
    USING (tenant_id = app_tenant() AND acl && app_principals());
CREATE POLICY documents_app_insert ON documents FOR INSERT TO app_user
    WITH CHECK (tenant_id = app_tenant() AND uploaded_by::text = app_user_id());

CREATE POLICY chunks_app_read ON chunks FOR SELECT TO app_user
    USING (tenant_id = app_tenant() AND acl && app_principals());

CREATE POLICY audit_app_insert ON retrieval_audit FOR INSERT TO app_user
    WITH CHECK (tenant_id = app_tenant());

-- Worker role (rag_worker): tenant only (it is a system process, not a user).
CREATE POLICY documents_worker_read ON documents FOR SELECT TO rag_worker
    USING (tenant_id = app_tenant());
CREATE POLICY documents_worker_update ON documents FOR UPDATE TO rag_worker
    USING (tenant_id = app_tenant()) WITH CHECK (tenant_id = app_tenant());
CREATE POLICY chunks_worker_all ON chunks FOR ALL TO rag_worker
    USING (tenant_id = app_tenant()) WITH CHECK (tenant_id = app_tenant());

-- ---------------------------------------------------------------- least-privilege grants
GRANT USAGE ON SCHEMA public TO app_user, rag_worker;

GRANT SELECT          ON tenants         TO app_user, rag_worker;
GRANT SELECT          ON users           TO app_user;
GRANT SELECT, INSERT  ON documents       TO app_user;
GRANT SELECT          ON chunks          TO app_user;          -- API can never write chunks
GRANT INSERT          ON ingest_jobs     TO app_user;
GRANT INSERT          ON retrieval_audit TO app_user;          -- no UPDATE/DELETE: append-only
GRANT USAGE ON SEQUENCE ingest_jobs_id_seq, retrieval_audit_id_seq TO app_user;

GRANT SELECT                            ON documents   TO rag_worker;
GRANT UPDATE (status, error, updated_at) ON documents  TO rag_worker;  -- cannot touch ACL
GRANT SELECT, INSERT, DELETE            ON chunks      TO rag_worker;
GRANT SELECT, UPDATE                    ON ingest_jobs TO rag_worker;
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS retrieval_audit, ingest_jobs, chunks, documents, users, tenants CASCADE;
DROP FUNCTION IF EXISTS chunks_inherit_acl(), app_principals(), app_user_id(), app_tenant();
"""


def _run_raw(sql: str) -> None:
    # Raw DB-API cursor with no params, so plpgsql's '%' placeholders aren't parsed by the driver.
    with op.get_bind().connection.dbapi_connection.cursor() as cur:
        cur.execute(sql)


def upgrade() -> None:
    _run_raw(UPGRADE_SQL)


def downgrade() -> None:
    _run_raw(DOWNGRADE_SQL)
