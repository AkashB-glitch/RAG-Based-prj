"""Week 6 Production API, Observability, Health Checks & Reliability Tests."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.observability import metrics


@pytest.mark.asyncio
async def test_health_liveness():
    """Verify /health returns 200 OK without requiring database connectivity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "permrag"
        assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_health_readiness_structure():
    """Verify /health/ready reports component readiness."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "database" in data
        assert "embedder" in data
        assert "reranker" in data


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Verify /metrics returns safe aggregated operational metrics without leaking documents."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests" in data
        assert "averages" in data
        assert "security" in data
        # Check security counters exist
        assert data["security"]["unauthorized_retrievals"] == 0
        assert data["security"]["cross_tenant_leaks"] == 0


@pytest.mark.asyncio
async def test_request_id_correlation():
    """Verify incoming X-Request-ID is preserved or generated and returned in headers."""
    transport = ASGITransport(app=app)
    custom_req_id = "test-custom-uuid-12345"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health", headers={"X-Request-ID": custom_req_id})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_req_id


@pytest.mark.asyncio
async def test_demo_ui_endpoint():
    """Verify / and /demo serve the interactive demonstration web UI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/demo")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "PermRAG" in resp.text
        assert "Alice" in resp.text


def test_rate_limiter_logic():
    """Verify in-memory sliding window rate limiter logic directly."""
    from app.middleware import InMemoryRateLimiter

    limiter = InMemoryRateLimiter()
    client = "192.168.1.100"

    # Allow up to 3 requests in 60s
    assert limiter.check_rate_limit(client, "test_cat", 3, window_seconds=60.0) is True
    assert limiter.check_rate_limit(client, "test_cat", 3, window_seconds=60.0) is True
    assert limiter.check_rate_limit(client, "test_cat", 3, window_seconds=60.0) is True
    # 4th request must be rejected
    assert limiter.check_rate_limit(client, "test_cat", 3, window_seconds=60.0) is False
    # Distinct client is not impacted
    assert limiter.check_rate_limit("192.168.1.101", "test_cat", 3, window_seconds=60.0) is True


@pytest.mark.asyncio
async def test_rate_limiting_middleware_429():
    """Verify rate limiter middleware triggers HTTP 429 when threshold is reached."""
    from app.config import settings
    from app.middleware import rate_limiter

    rate_limiter._history.clear()
    original_limit = settings.rate_limit_rag_per_minute
    settings.rate_limit_rag_per_minute = 2

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1st request -> passes rate limit middleware (even if auth fails with 401/403)
            r1 = await client.post("/rag/query", json={"query": "test"})
            assert r1.status_code != 429

            # 2nd request -> passes rate limit middleware
            r2 = await client.post("/rag/query", json={"query": "test"})
            assert r2.status_code != 429

            # 3rd request -> blocked by rate limit middleware
            r3 = await client.post("/rag/query", json={"query": "test"})
            assert r3.status_code == 429
            assert "RAG query rate limit exceeded" in r3.json()["error"]
    finally:
        settings.rate_limit_rag_per_minute = original_limit
        rate_limiter._history.clear()

