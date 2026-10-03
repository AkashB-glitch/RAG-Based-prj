"""Adversarial tests for Session Isolation, Connection Pooling, and Concurrency."""
import asyncio
import psycopg
import pytest

from app.principals import AuthContext
from app.retrieval.scoring import SearchRequest
from app.retrieval.service import retrieval_service
from tests.conftest import APP_DSN, as_role, requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


def test_attack_connection_pool_residual_context(retrieval_world):
    """Test connection reuse: Request A sets Tenant A; completes.

    Next query on the SAME raw connection must NOT inherit Tenant A context.
    """
    tenant_a = retrieval_world["tenants"]["a"]

    with psycopg.connect(APP_DSN) as conn:
        # 1. Transaction 1: Alice operates with Tenant A context (is_local=true)
        with conn.transaction():
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_a,))
            conn.execute("SELECT set_config('app.principals', 'role:engineer', true)")
            rows_a = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            assert rows_a > 0

        # Transaction 1 committed.
        # 2. Transaction 2: New query without setting context on the same connection
        with conn.transaction():
            # Current setting should be empty/cleared by PostgreSQL
            curr_tenant = conn.execute("SELECT nullif(current_setting('app.tenant_id', true), '')").fetchone()[0]
            curr_principals = conn.execute("SELECT nullif(current_setting('app.principals', true), '')").fetchone()[0]
            rows_unauth = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    # Must be completely cleared (fail closed)
    assert curr_tenant is None
    assert curr_principals is None
    assert rows_unauth == 0


@pytest.mark.asyncio
async def test_attack_concurrent_multi_tenant_queries_do_not_bleed(retrieval_world):
    """Run concurrent queries from Tenant A and Tenant B simultaneously.

    Asserts zero cross-tenant contamination under asynchronous concurrency.
    """
    tenant_a = retrieval_world["tenants"]["a"]
    tenant_b = retrieval_world["tenants"]["b"]

    ctx_a = AuthContext(tenant_id=tenant_a, user_id="alice", roles=("engineer",), groups=())
    ctx_b = AuthContext(tenant_id=tenant_b, user_id="charlie", roles=("engineer",), groups=())

    req_a = SearchRequest(query="engineering deployment procedure", top_k=5)
    req_b = SearchRequest(query="engineering deployment procedure", top_k=5)

    # Execute 20 concurrent requests interleaved across Tenant A and Tenant B
    tasks = []
    for i in range(10):
        tasks.append(retrieval_service.search(request=req_a, ctx=ctx_a))
        tasks.append(retrieval_service.search(request=req_b, ctx=ctx_b))

    results = await asyncio.gather(*tasks)

    for idx, resp in enumerate(results):
        expected_tenant = "tenant a" if (idx % 2 == 0) else "tenant b"
        forbidden_tenant = "tenant b" if (idx % 2 == 0) else "tenant a"

        for r in resp.results:
            assert forbidden_tenant not in r.content.lower(), f"Leak in task {idx}: found {forbidden_tenant}"
            assert expected_tenant in r.content.lower(), f"Missing expected content in task {idx}"
