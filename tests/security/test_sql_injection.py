"""Adversarial tests for SQL Injection in Search and Document Operations."""
import pytest

from app.principals import AuthContext
from app.retrieval.keyword import search_keyword
from app.retrieval.scoring import SearchRequest
from app.retrieval.service import retrieval_service
from tests.conftest import requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql_payload",
    [
        "' OR 1=1; --",
        "'; DROP TABLE documents; --",
        "' UNION SELECT NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL; --",
        "admin'--",
        "' OR ''='",
        "1; SELECT pg_sleep(1); --",
        "'; UPDATE chunks SET content = 'hacked'; --",
    ],
)
async def test_attack_sql_injection_in_search_query(retrieval_world, sql_payload):
    """Adversarial SQL injection payloads in search queries must be safely parameterized."""
    tenant_a = retrieval_world["tenants"]["a"]
    alice_ctx = AuthContext(
        tenant_id=tenant_a,
        user_id="alice",
        roles=("engineer",),
        groups=(),
    )

    # Hybrid Search
    req = SearchRequest(query=sql_payload, top_k=5, method="hybrid")
    resp = await retrieval_service.search(request=req, ctx=alice_ctx)
    # Must execute safely without syntax errors or table manipulation
    assert isinstance(resp.results, list)

    # Full-Text Search specifically
    req_kw = SearchRequest(query=sql_payload, top_k=5, method="keyword")
    resp_kw = await retrieval_service.search(request=req_kw, ctx=alice_ctx)
    assert isinstance(resp_kw.results, list)
