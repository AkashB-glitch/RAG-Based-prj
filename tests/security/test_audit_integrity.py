"""Adversarial tests for Audit Log Immutability and Completeness."""
import uuid
import psycopg
import pytest

from app.principals import AuthContext
from app.rag.schemas import RAGRequest
from app.rag.service import rag_service
from tests.conftest import APP_DSN, OWNER_DSN, as_role, requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


def test_attack_unauthorized_audit_log_deletion(retrieval_world):
    """An attacker connecting as app_user attempts to delete audit logs."""
    tenant_a = retrieval_world["tenants"]["a"]
    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM retrieval_audit")


def test_attack_unauthorized_audit_log_tampering(retrieval_world):
    """An attacker connecting as app_user attempts to modify existing audit records."""
    tenant_a = retrieval_world["tenants"]["a"]
    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("UPDATE retrieval_audit SET user_id = 'compromised'")


def test_attack_cross_tenant_audit_forgery(retrieval_world):
    """An attacker with Tenant A context attempts to insert an audit record attributing activity to Tenant B."""
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b = retrieval_world["tenants"]["b"]

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                """
                INSERT INTO retrieval_audit (
                    request_id, tenant_id, user_id, principals, query_sha256, chunk_ids, document_ids
                ) VALUES (
                    gen_random_uuid(), %s::uuid, 'alice', '{role:engineer}', 'hash', '{}', '{}'
                )
                """,
                (tenant_b,),
            )


@pytest.mark.asyncio
async def test_audit_completeness_traceable_pipeline(retrieval_world):
    """Verifies every RAG request leaves a complete, immutable audit record."""
    tenant_a = retrieval_world["tenants"]["a"]
    user_id = f"audited-{uuid.uuid4().hex[:6]}"
    req_id = uuid.uuid4()

    ctx = AuthContext(
        tenant_id=tenant_a,
        user_id=user_id,
        roles=("engineer",),
        groups=(),
    )

    req = RAGRequest(query="engineering deployment procedure", top_k=3)
    await rag_service.query(request=req, ctx=ctx, request_id=req_id)

    with psycopg.connect(OWNER_DSN) as conn:
        row = conn.execute(
            """
            SELECT request_id::text, tenant_id::text, user_id, query_sha256, chunk_ids, document_ids, latency_ms
              FROM retrieval_audit
             WHERE request_id = %s::uuid
            """,
            (str(req_id),),
        ).mappings().first()

    assert row is not None
    assert row["request_id"] == str(req_id)
    assert row["tenant_id"] == tenant_a
    assert row["user_id"] == user_id
    assert len(row["query_sha256"]) == 64  # valid SHA-256
    assert isinstance(row["chunk_ids"], list)
    assert isinstance(row["document_ids"], list)
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0
