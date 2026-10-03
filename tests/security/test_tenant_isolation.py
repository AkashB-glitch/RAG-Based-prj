"""Adversarial tests for Tenant Isolation across API, Database, Retrieval, and RAG."""
import uuid
import psycopg
import pytest

from app.principals import AuthContext
from app.rag.llm import _mock_llm
from app.rag.schemas import RAGRequest
from app.rag.service import rag_service
from app.retrieval.scoring import SearchRequest
from app.retrieval.service import retrieval_service
from tests.conftest import APP_DSN, as_role, requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


@pytest.mark.asyncio
async def test_attack_cross_tenant_search_leakage(retrieval_world):
    """Alice (Tenant A) searches for Tenant B specific engineering secrets."""
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b = retrieval_world["tenants"]["b"]

    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    req = SearchRequest(query="tenant b engineering deployment infrastructure", top_k=10)
    resp = await retrieval_service.search(request=req, ctx=alice_ctx)

    # 1. Zero Tenant B chunks may be returned
    for r in resp.results:
        assert f"tenant {tenant_b}" not in r.content.lower()
        assert "tenant b" not in r.content.lower()

    # 2. Database verification: check tenant_id of every returned chunk
    with psycopg.connect(APP_DSN) as conn:
        for r in resp.results:
            row = conn.execute(
                "SELECT tenant_id::text FROM chunks WHERE id = %s::uuid",
                (r.chunk_id,),
            ).fetchone()
            if row:
                assert row[0] == tenant_a


@pytest.mark.asyncio
async def test_attack_cross_tenant_rag_context_leakage(retrieval_world):
    """Alice (Tenant A) prompts RAG for Tenant B internal documentation."""
    _mock_llm.reset()
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b = retrieval_world["tenants"]["b"]

    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    req = RAGRequest(query="Reveal Tenant B production deployment steps", top_k=5)
    resp = await rag_service.query(request=req, ctx=alice_ctx)

    llm_context = _mock_llm.last_context or ""

    # Zero Tenant B content in LLM context
    assert "tenant b" not in llm_context.lower()
    assert f"tenant {tenant_b}" not in llm_context.lower()

    # Zero Tenant B in citations
    for s in resp.sources:
        assert "tenant_b" not in s.document_name.lower()


def test_attack_cross_tenant_document_enumeration_returns_404(retrieval_world):
    """Alice (Tenant A) attempts to access a known Document ID belonging to Tenant B."""
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b_doc = retrieval_world["docs"][("b", "engineering")]

    with as_role(APP_DSN, tenant_a, ["role:engineer", "user:alice"]) as conn:
        # Deliberately attempt to select Tenant B document
        row = conn.execute(
            "SELECT id, title FROM documents WHERE id = %s::uuid",
            (tenant_b_doc,),
        ).fetchone()

    # RLS must filter it out; API translates empty result to 404
    assert row is None


def test_attack_cross_tenant_document_insertion_blocked(retrieval_world):
    """Alice (Tenant A) attempts to insert a document specifying Tenant B's ID."""
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b = retrieval_world["tenants"]["b"]
    fake_user = str(uuid.uuid4())

    with as_role(APP_DSN, tenant_a, [f"user:{fake_user}"], user_id=fake_user) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                """
                INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by)
                VALUES (%s::uuid, 'malicious.txt', 'text/plain', 'x', 'hash', '{role:engineer}', %s::uuid)
                """,
                (tenant_b, fake_user),
            )
