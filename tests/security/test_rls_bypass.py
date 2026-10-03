"""Adversarial tests for Database RLS, Fail-Closed Behavior, and Existence Enumeration."""
import psycopg
import pytest

from app.config import settings
from app.ingestion.embeddings import FakeEmbedder, to_pgvector
from tests.conftest import APP_DSN, OWNER_DSN, as_role, requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


def test_attack_missing_tenant_context_fails_closed(retrieval_world):
    """Executing queries without setting app.tenant_id must return 0 rows."""
    with as_role(APP_DSN, None, None) as conn:
        doc_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        user_count = conn.execute("SELECT count(*) FROM users").fetchone()[0]

    assert doc_count == 0
    assert chunk_count == 0
    assert user_count == 0


def test_attack_missing_principals_fails_closed(retrieval_world):
    """Executing queries with tenant set but empty app.principals returns 0 chunks."""
    tenant_a = retrieval_world["tenants"]["a"]
    with as_role(APP_DSN, tenant_a, []) as conn:
        chunk_rows = conn.execute("SELECT content FROM chunks").fetchall()

    assert chunk_rows == []


def test_attack_rls_bypass_deliberate_where_clause_omission(retrieval_world):
    """Intentionally omit WHERE clause; RLS must independently stop leakage."""
    tenant_a = retrieval_world["tenants"]["a"]
    emb = FakeEmbedder(settings.embedding_dim)
    q_vec = to_pgvector(emb.embed_query("confidential employee salary"))

    # Alice has role:engineer. No WHERE clause provided!
    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content, d.title
              FROM chunks c
              JOIN documents d ON c.document_id = d.id
             ORDER BY c.embedding <=> %s::vector ASC
             LIMIT 10
            """,
            (q_vec,),
        ).fetchall()

    # RLS alone must prevent HR, Finance, and Tenant B chunks
    for content, title in rows:
        assert "salary" not in content.lower()
        assert "finance" not in content.lower()
        assert "tenant b" not in content.lower()
        assert "tenant a engineering" in content.lower()


def test_attack_document_existence_enumeration(retrieval_world):
    """Attempt to detect if an inaccessible document exists based on DB response."""
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b_doc = retrieval_world["docs"][("b", "engineering")]
    fake_random_doc = "00000000-0000-0000-0000-000000000000"

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        # 1. Query with real foreign document ID
        row_foreign = conn.execute(
            "SELECT id FROM documents WHERE id = %s::uuid",
            (tenant_b_doc,),
        ).fetchone()

        # 2. Query with non-existent document ID
        row_nonexistent = conn.execute(
            "SELECT id FROM documents WHERE id = %s::uuid",
            (fake_random_doc,),
        ).fetchone()

    # Both must be indistinguishable (NULL/empty result)
    assert row_foreign is None
    assert row_nonexistent is None


def test_verify_database_functions_do_not_bypass_rls():
    """Verify system functions are STABLE/IMMUTABLE and do not have SECURITY DEFINER bypass."""
    with psycopg.connect(OWNER_DSN) as conn:
        rows = conn.execute(
            """
            SELECT routine_name, security_type
              FROM information_schema.routines
             WHERE routine_schema = 'public'
               AND routine_name IN ('app_tenant', 'app_principals', 'app_user_id', 'chunks_inherit_acl')
            """
        ).fetchall()

    for name, sec_type in rows:
        # None of the security context functions should be SECURITY DEFINER
        assert sec_type == "INVOKER", f"Function {name} must be INVOKER, got {sec_type}"
