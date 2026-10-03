"""RAG Service for PermRAG: Secure retrieval-augmented generation and citation linking."""
import logging
import time
import uuid

from app.principals import AuthContext
from app.rag.citations import build_citations
from app.rag.context import context_builder
from app.rag.llm import get_llm_provider
from app.rag.prompts import NO_AUTHORIZED_INFO_MESSAGE, SYSTEM_PROMPT
from app.rag.schemas import LatencyBreakdown, RAGRequest, RAGResponse
from app.retrieval.scoring import SearchRequest
from app.retrieval.service import retrieval_service

log = logging.getLogger("rag")


class RAGService:
    def __init__(self):
        self._llm = get_llm_provider()

    async def query(
        self,
        request: RAGRequest,
        ctx: AuthContext,
        request_id: uuid.UUID | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        req_id = request_id or uuid.uuid4()
        clean_query = request.query.strip()

        latency = LatencyBreakdown()

        if not clean_query:
            return RAGResponse(
                answer=NO_AUTHORIZED_INFO_MESSAGE,
                sources=[],
                latency=latency,
            )

        # 1. Permission-aware retrieval under user's database RLS context
        retrieval_start = time.perf_counter()
        retrieval_resp = await retrieval_service.search(
            request=SearchRequest(
                query=clean_query,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                rerank=request.rerank,
                method="hybrid",
            ),
            ctx=ctx,
            request_id=req_id,
            record_audit=True,
        )
        latency.retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # 2. Check for empty authorized results (Fail-Closed / Existence Privacy)
        if not retrieval_resp.results:
            latency.total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            return RAGResponse(
                answer=NO_AUTHORIZED_INFO_MESSAGE,
                sources=[],
                latency=latency,
            )

        # 3. Context Construction with token budgeting
        context_start = time.perf_counter()
        formatted_context, included_results = context_builder.build_context(retrieval_resp.results)
        latency.context_ms = round((time.perf_counter() - context_start) * 1000, 2)

        # 4. LLM Generation
        llm_start = time.perf_counter()
        provider = get_llm_provider()
        answer = await provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_query=clean_query,
            context=formatted_context,
        )
        latency.llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

        # 5. Source Citations
        citations = build_citations(included_results)

        latency.total_ms = round((time.perf_counter() - total_start) * 1000, 2)

        log.info(
            "rag_query request_id=%s tenant=%s user=%s retrieved=%d context=%d latency_ms=%.1f",
            req_id,
            ctx.tenant_id,
            ctx.user_id,
            len(retrieval_resp.results),
            len(included_results),
            latency.total_ms,
        )

        return RAGResponse(
            answer=answer,
            sources=citations,
            latency=latency,
        )


rag_service = RAGService()
