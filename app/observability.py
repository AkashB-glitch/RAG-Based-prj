"""Observability, structured logging, request tracing, and metrics for PermRAG."""
import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

from app.config import settings

# Request context correlation
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
tenant_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="")
user_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")


def get_current_request_id() -> str:
    req_id = request_id_ctx.get()
    return req_id if req_id else str(uuid.uuid4())


class JSONLogFormatter(logging.Formatter):
    """Structured JSON formatter ensuring zero leakage of secrets or raw documents."""

    # Never log fields matching these keys
    SENSITIVE_KEYS = {"password", "jwt_secret", "secret", "token", "authorization", "api_key", "raw", "raw_prompt"}

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get() or None,
            "tenant_id": tenant_id_ctx.get() or None,
            "user_id": user_id_ctx.get() or None,
        }

        # Include custom extra fields if not sensitive
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            for k, v in record.extra.items():
                if k.lower() not in self.SENSITIVE_KEYS:
                    log_obj[k] = v

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging():
    """Initializes structured logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if settings.enable_structured_logging and settings.env != "test":
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s"))

    root.addHandler(handler)


class MetricsRegistry:
    """Aggregated operational metrics registry.

    Invariant: Does NOT log or expose sensitive high-cardinality document contents or queries.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Latencies
        self.total_latency_ms = 0.0
        self.retrieval_latency_ms = 0.0
        self.rerank_latency_ms = 0.0
        self.context_latency_ms = 0.0
        self.llm_latency_ms = 0.0

        # Chunks and tokens
        self.total_retrieved_chunks = 0
        self.total_reranked_chunks = 0
        self.total_final_chunks = 0
        self.total_context_tokens = 0

        # Security counters (must always remain 0)
        self.unauthorized_retrieval_attempts = 0
        self.unauthorized_reranker_attempts = 0
        self.unauthorized_llm_context_attempts = 0
        self.cross_tenant_attempts = 0
        self.cross_role_attempts = 0

    def record_rag_request(
        self,
        retrieval_ms: float,
        rerank_ms: float,
        context_ms: float,
        llm_ms: float,
        total_ms: float,
        retrieved_count: int,
        final_count: int,
        context_tokens: int,
        success: bool = True,
    ):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.retrieval_latency_ms += retrieval_ms
        self.rerank_latency_ms += rerank_ms
        self.context_latency_ms += context_ms
        self.llm_latency_ms += llm_ms
        self.total_latency_ms += total_ms

        self.total_retrieved_chunks += retrieved_count
        self.total_final_chunks += final_count
        self.total_context_tokens += context_tokens

    def get_summary(self) -> dict[str, Any]:
        req_count = max(1, self.total_requests)
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
            },
            "averages": {
                "avg_total_latency_ms": round(self.total_latency_ms / req_count, 2),
                "avg_retrieval_latency_ms": round(self.retrieval_latency_ms / req_count, 2),
                "avg_rerank_latency_ms": round(self.rerank_latency_ms / req_count, 2),
                "avg_llm_latency_ms": round(self.llm_latency_ms / req_count, 2),
                "avg_retrieved_chunks": round(self.total_retrieved_chunks / req_count, 1),
                "avg_final_chunks": round(self.total_final_chunks / req_count, 1),
                "avg_context_tokens": round(self.total_context_tokens / req_count, 1),
            },
            "security": {
                "unauthorized_retrievals": self.unauthorized_retrieval_attempts,
                "unauthorized_reranker_inputs": self.unauthorized_reranker_attempts,
                "unauthorized_context_inclusions": self.unauthorized_llm_context_attempts,
                "cross_tenant_leaks": self.cross_tenant_attempts,
                "cross_role_leaks": self.cross_role_attempts,
            },
        }


metrics = MetricsRegistry()
