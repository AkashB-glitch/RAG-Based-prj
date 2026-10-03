"""Adversarial tests for Role/Group ACL Isolation."""
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
async def test_attack_engineer_attempts_salary_retrieval(retrieval_world):
    """Alice (engineer) searches for confidential HR compensation policies."""
    tenant_a = retrieval_world["tenants"]["a"]
    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    for method in ("semantic", "keyword", "hybrid"):
        req = SearchRequest(
            query="confidential employee salary bands merit bonus compensation",
            top_k=5,
            method=method,
        )
        resp = await retrieval_service.search(request=req, ctx=alice_ctx)
        for r in resp.results:
            assert "salary policy" not in r.content.lower()
            assert "merit bonus" not in r.content.lower()
            assert "compensation bands" not in r.content.lower()


@pytest.mark.asyncio
async def test_attack_engineer_attempts_finance_audit_retrieval(retrieval_world):
    """Alice (engineer) searches for finance audit and quarterly budget records."""
    tenant_a = retrieval_world["tenants"]["a"]
    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    req = SearchRequest(
        query="quarterly financial budget audit statements tax report",
        top_k=5,
        method="hybrid",
    )
    resp = await retrieval_service.search(request=req, ctx=alice_ctx)
    for r in resp.results:
        assert "financial budget" not in r.content.lower()
        assert "tax report" not in r.content.lower()


@pytest.mark.asyncio
async def test_attack_engineer_attempts_rag_salary_disclosure(retrieval_world):
    """Alice (engineer) asks RAG to disclose executive compensation."""
    _mock_llm.reset()
    tenant_a = retrieval_world["tenants"]["a"]
    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    req = RAGRequest(query="What is the confidential executive salary and bonus pool?", top_k=5)
    resp = await rag_service.query(request=req, ctx=alice_ctx)

    llm_context = _mock_llm.last_context or ""
    # Invariant: UNAUTHORIZED_CHUNKS ∩ LLM_CONTEXT = ∅
    assert "salary policy" not in llm_context.lower()
    assert "merit bonus" not in llm_context.lower()
    assert "compensation bands" not in llm_context.lower()

    # Citations must not leak
    for s in resp.sources:
        assert "salary" not in s.document_name.lower()
        assert "finance" not in s.document_name.lower()


def test_authorized_hr_user_can_access_salary_data(retrieval_world):
    """Bob (role:hr) should legitimately access HR salary documents."""
    tenant_a = retrieval_world["tenants"]["a"]

    with as_role(APP_DSN, tenant_a, ["role:hr", "user:bob"]) as conn:
        rows = conn.execute(
            """
            SELECT c.content, d.title
              FROM chunks c
              JOIN documents d ON c.document_id = d.id
             WHERE c.tsv @@ websearch_to_tsquery('english', 'confidential salary')
            """
        ).fetchall()

    assert len(rows) > 0
    assert any("confidential employee salary" in r[0] for r in rows)
