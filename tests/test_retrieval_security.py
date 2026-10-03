"""Security verification suite for Week 2 permission-aware retrieval engine.

Tests all 8 required security scenarios:
1. Authorized retrieval
2. Same tenant but unauthorized role
3. Cross-tenant isolation
4. Missing security context (fail-closed)
5. Query without application WHERE clauses (RLS alone prevents leakage)
6. Highly relevant unauthorized document (cannot be returned even if similarity is higher)
7. Full-text search authorization under RLS
8. Hybrid search authorization under RLS
"""
import uuid
import psycopg
import pytest

from app.config import settings
from app.ingestion.embeddings import FakeEmbedder, to_pgvector
from tests.conftest import APP_DSN, OWNER_DSN, as_role, requires_db

pytestmark = requires_db


@pytest.fixture
def retrieval_world():
    """Sets up two isolated tenants with Engineering, HR, and Finance documents and vector embeddings."""
    emb = FakeEmbedder(settings.embedding_dim)
    w = {
        "tenants": {},
        "docs": {},
        "chunks": {},
    }

    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        for tenant_key in ("a", "b"):
            slug = f"retrieval-{tenant_key}-{uuid.uuid4().hex[:8]}"
            t_id = str(conn.execute("INSERT INTO tenants (slug, name) VALUES (%s, %s) RETURNING id", (slug, slug)).fetchone()[0])
            w["tenants"][tenant_key] = t_id

            doc_specs = [
                (
                    "engineering",
                    ["role:engineer"],
                    f"tenant {tenant_key} engineering deployment procedure for kubernetes cluster in production",
                    f"tenant_{tenant_key}_engineering.pdf",
                ),
                (
                    "hr",
                    ["role:hr"],
                    f"tenant {tenant_key} confidential employee salary policy compensation bands and merit bonus",
                    f"tenant_{tenant_key}_salary_policy.pdf",
                ),
                (
                    "finance",
                    ["role:finance"],
                    f"tenant {tenant_key} quarterly financial budget audit statements and tax report",
                    f"tenant_{tenant_key}_finance_audit.pdf",
                ),
            ]

            for doc_key, acl, content, title in doc_specs:
                d_id = str(
                    conn.execute(
                        """
                        INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by)
                        VALUES (%s, %s, 'text/plain', %s, 'x', %s, gen_random_uuid()) RETURNING id
                        """,
                        (t_id, title, content.encode("utf-8"), acl),
                    ).fetchone()[0]
                )
                w["docs"][(tenant_key, doc_key)] = d_id

                c_id = str(
                    conn.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, 0, %s, %s::vector) RETURNING id
                        """,
                        (d_id, content, to_pgvector(emb.embed_query(content))),
                    ).fetchone()[0]
                )
                w["chunks"][(tenant_key, doc_key)] = c_id

    yield w

    # Teardown
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM tenants WHERE id = ANY(%s::uuid[])", (list(w["tenants"].values()),))


# -----------------------------------------------------------------------------
# Test 1: Authorized retrieval
# -----------------------------------------------------------------------------
def test_authorized_retrieval_success(retrieval_world):
    """Alice (Tenant A, engineer) searches for engineering deployment procedure."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("engineering deployment procedure"))

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        # Layer 1 + Layer 2 query
        rows = conn.execute(
            """
            SELECT c.content, d.title
              FROM chunks c
              JOIN documents d ON c.document_id = d.id
             WHERE c.tenant_id = %s::uuid AND c.acl && %s::text[]
             ORDER BY c.embedding <=> %s::vector ASC
             LIMIT 5
            """,
            (tenant_a, ["role:engineer", "user:alice"], q_vec),
        ).fetchall()

    assert len(rows) > 0
    assert any("tenant a engineering deployment" in r[0] for r in rows)
    assert not any("salary" in r[0] for r in rows)
    assert not any("finance" in r[0] for r in rows)


# -----------------------------------------------------------------------------
# Test 2: Same tenant but unauthorized role
# -----------------------------------------------------------------------------
def test_same_tenant_unauthorized_role_blocked(retrieval_world):
    """Alice (Tenant A, engineer) searches for employee salary policy; HR docs must NOT appear."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("employee salary policy compensation"))

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content
              FROM chunks c
             WHERE c.tenant_id = %s::uuid AND c.acl && %s::text[]
             ORDER BY c.embedding <=> %s::vector ASC
             LIMIT 5
            """,
            (tenant_a, ["role:engineer", "user:alice"], q_vec),
        ).fetchall()

    contents = [r[0] for r in rows]
    # Zero HR or finance documents may appear
    assert not any("salary" in c for c in contents)
    assert not any("confidential" in c for c in contents)
    assert not any("finance" in c for c in contents)


# -----------------------------------------------------------------------------
# Test 3: Cross-tenant isolation
# -----------------------------------------------------------------------------
def test_cross_tenant_isolation(retrieval_world):
    """Alice (Tenant A) searches for Tenant B information. Zero Tenant B chunks returned."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("tenant b engineering deployment"))

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content, c.tenant_id::text
              FROM chunks c
             ORDER BY c.embedding <=> %s::vector ASC
             LIMIT 10
            """,
            (q_vec,),
        ).fetchall()

    contents = [r[0] for r in rows]
    tenant_ids = [r[1] for r in rows]

    # Alice can only see Tenant A rows
    assert not any("tenant b" in c for c in contents)
    assert all(t == tenant_a for t in tenant_ids)


# -----------------------------------------------------------------------------
# Test 4: Missing security context (fail-closed)
# -----------------------------------------------------------------------------
def test_missing_security_context_returns_zero_rows(retrieval_world):
    """Executing retrieval without setting tenant/principal context must return 0 rows."""
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("deployment procedure"))

    # Connect as app_user with NULL context
    with as_role(APP_DSN, None, None) as conn:
        rows = conn.execute(
            "SELECT c.id FROM chunks c ORDER BY c.embedding <=> %s::vector ASC LIMIT 10",
            (q_vec,),
        ).fetchall()
        assert rows == []

        # Full text search with missing context
        kw_rows = conn.execute(
            "SELECT c.id FROM chunks c WHERE c.tsv @@ websearch_to_tsquery('english', 'deployment') LIMIT 10"
        ).fetchall()
        assert kw_rows == []


# -----------------------------------------------------------------------------
# Test 5: Query without application WHERE clauses (RLS alone prevents leakage)
# -----------------------------------------------------------------------------
def test_retrieval_without_application_where_clause(retrieval_world):
    """Deliberately omit application-level WHERE filters. PostgreSQL RLS alone must block leaks."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("confidential employee salary"))

    # Alice has role:engineer. Deliberately no WHERE clause!
    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        rows = conn.execute(
            "SELECT c.content FROM chunks c ORDER BY c.embedding <=> %s::vector ASC LIMIT 10",
            (q_vec,),
        ).fetchall()

    contents = [r[0] for r in rows]
    # RLS alone must prevent HR / Finance / Tenant B docs from showing up
    assert not any("salary" in c for c in contents)
    assert not any("finance" in c for c in contents)
    assert not any("tenant b" in c for c in contents)


# -----------------------------------------------------------------------------
# Test 6: Highly relevant unauthorized document
# -----------------------------------------------------------------------------
def test_highly_relevant_unauthorized_document_is_never_returned(retrieval_world):
    """An unauthorized document has near-perfect semantic match to the query.

    PostgreSQL RLS must ensure it is NOT returned, even when it would otherwise rank #1.
    """
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)

    # Search query perfectly matching the unauthorized HR document
    query = "confidential employee salary policy compensation bands and merit bonus"
    q_vec = to_pgvector(emb.embed_query(query))

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        # Deliberately search with NO WHERE clause: order by vector distance
        rows = conn.execute(
            """
            SELECT c.content, (1 - (c.embedding <=> %s::vector)) AS similarity
              FROM chunks c
             ORDER BY c.embedding <=> %s::vector ASC
             LIMIT 5
            """,
            (q_vec, q_vec),
        ).fetchall()

    # The unauthorized chunk has ~1.0 similarity, but RLS drops it.
    # Alice only gets her authorized engineering chunk (or nothing if threshold applied).
    for content, sim in rows:
        assert "salary" not in content
        assert "confidential" not in content
        assert "tenant a engineering" in content


# -----------------------------------------------------------------------------
# Test 7: Full-text authorization
# -----------------------------------------------------------------------------
def test_fulltext_search_authorization(retrieval_world):
    """Verify full-text search respects RLS boundaries."""
    tenant_a = retrieval_world["tenants"]["a"]

    # Bob has role:hr. Searches for "confidential salary"
    with as_role(APP_DSN, tenant_a, ["role:hr", "user:bob"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content
              FROM chunks c
             WHERE c.tsv @@ websearch_to_tsquery('english', 'confidential salary')
            """
        ).fetchall()
        assert len(rows) > 0
        assert all("tenant a confidential employee salary" in r[0] for r in rows)

    # Alice has role:engineer. Searches for "confidential salary"
    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content
              FROM chunks c
             WHERE c.tsv @@ websearch_to_tsquery('english', 'confidential salary')
            """
        ).fetchall()
        # Even without app WHERE clause, RLS yields 0 rows for Alice
        assert rows == []


# -----------------------------------------------------------------------------
# Test 8: Hybrid authorization
# -----------------------------------------------------------------------------
def test_hybrid_search_authorization(retrieval_world):
    """Verify hybrid search (vector + keyword) preserves zero unauthorized retrieval."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("budget audit and deployment"))

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        # Semantic search without WHERE clause
        sem_rows = conn.execute(
            "SELECT c.content FROM chunks c ORDER BY c.embedding <=> %s::vector ASC LIMIT 5",
            (q_vec,),
        ).fetchall()

        # Keyword search without WHERE clause
        kw_rows = conn.execute(
            "SELECT c.content FROM chunks c WHERE c.tsv @@ websearch_to_tsquery('english', 'budget audit deployment') LIMIT 5"
        ).fetchall()

    all_returned = [r[0] for r in sem_rows] + [r[0] for r in kw_rows]
    # No finance or HR content leaked to engineer
    assert not any("budget" in c for c in all_returned)
    assert not any("salary" in c for c in all_returned)
    assert not any("tenant b" in c for c in all_returned)
    # Only tenant A engineering could be returned
    assert all("tenant a engineering" in c for c in all_returned)
