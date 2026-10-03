"""Production middleware: request correlation, rate limiting, and safe error handling."""
import logging
import time
import uuid
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.observability import request_id_ctx

log = logging.getLogger("middleware")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Correlates each incoming request with a unique Request ID across the entire lifecycle."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = req_id

            # Safe access logging (does not log request body or sensitive headers)
            if request.url.path not in ("/health", "/health/ready"):
                log.info(
                    "request completed path=%s method=%s status=%d duration_ms=%.2f req_id=%s",
                    request.url.path,
                    request.method,
                    response.status_code,
                    duration_ms,
                    req_id,
                )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log.exception(
                "unhandled exception in request path=%s method=%s duration_ms=%.2f req_id=%s: %s",
                request.url.path,
                request.method,
                duration_ms,
                req_id,
                str(exc),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": req_id,
                },
                headers={"X-Request-ID": req_id},
            )
        finally:
            request_id_ctx.reset(token)


class InMemoryRateLimiter:
    """Sliding-window rate limiter for sensitive endpoints.

    Guards against brute force on auth, spamming uploads, or denial-of-service on LLM queries.
    """

    def __init__(self):
        # Maps (client_key, endpoint_category) -> list of timestamps
        self._history: dict[tuple[str, str], list[float]] = defaultdict(list)

    def check_rate_limit(self, client_id: str, category: str, max_requests: int, window_seconds: float = 60.0) -> bool:
        now = time.time()
        key = (client_id, category)
        history = self._history[key]

        # Prune timestamps older than window
        cutoff = now - window_seconds
        while history and history[0] < cutoff:
            history.pop(0)

        if len(history) >= max_requests:
            return False

        history.append(now)
        return True


rate_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces configurable rate limits across endpoints."""

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        if path.startswith("/auth") and settings.rate_limit_auth_per_minute > 0:
            if not rate_limiter.check_rate_limit(client_ip, "auth", settings.rate_limit_auth_per_minute):
                return JSONResponse(status_code=429, content={"error": "Too many authentication attempts. Please retry later."})

        elif path.startswith("/documents") and request.method == "POST" and settings.rate_limit_upload_per_minute > 0:
            if not rate_limiter.check_rate_limit(client_ip, "upload", settings.rate_limit_upload_per_minute):
                return JSONResponse(status_code=429, content={"error": "Upload rate limit exceeded. Please retry later."})

        elif path.startswith("/rag") and settings.rate_limit_rag_per_minute > 0:
            if not rate_limiter.check_rate_limit(client_ip, "rag", settings.rate_limit_rag_per_minute):
                return JSONResponse(status_code=429, content={"error": "RAG query rate limit exceeded. Please retry later."})

        return await call_next(request)
