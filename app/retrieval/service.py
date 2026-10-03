import asyncio
import hashlib
import time
import uuid
import logging
from sqlalchemy import text

from app.config import settings
from app.db import tenant_tx
from app.ingestion.embeddings import get_embedder
from app.principals import AuthContext
from app.retrieval.dedup import deduplicate_and_diversify
from app.retrieval.hybrid import search_hybrid
from app.retrieval.keyword import search_keyword
from app.retrieval.query_expansion import expand_query
from app.retrieval.reranker import get_reranker
from app.retrieval.scoring import RetrievalResult, SearchRequest, SearchResponse
from app.retrieval.semantic import search_semantic

log = logging.getLogger("retrieval")


class RetrievalService:
    """Orchestrates permission-aware retrieval across semantic, keyword, and hybrid methods.

    Guarantees:
    - Never bypasses PostgreSQL RLS security boundaries.
    - Operates inside tenant transaction context (`tenant_tx`).
    - Enforces RERANKER_INPUT ⊆ AUTHORIZED_RETRIEVAL_RESULTS.
    - Records retrieval audit entry in the same transaction.
    - Never discloses existence of unauthorized documents.
    """

    def __init__(self):
        self._embedder = get_embedder()

    async def search(
        self,
        request: SearchRequest,
        ctx: AuthContext,
        request_id: uuid.UUID | None = None,
        record_audit: bool = True,
    ) -> SearchResponse:
        start_time = time.perf_counter()
        req_id = request_id or uuid.uuid4()
        top_k = request.top_k or settings.retrieval_top_k
        query = request.query.strip()

        if not query:
            return SearchResponse(query="", results=[], total_count=0)

        should_rerank = (request.rerank is True) or (request.rerank is None and settings.enable_reranking)
        cand_k = (request.candidate_k or settings.candidate_k) if should_rerank else top_k

        # Generate query embedding if semantic or hybrid is requested
        query_vector: list[float] | None = None
        if request.method in ("semantic", "hybrid"):
            query_vector = await asyncio.to_thread(self._embedder.embed_query, query)

        queries_to_search = [query]
        if settings.enable_query_expansion and request.method in ("keyword", "hybrid"):
            queries_to_search = expand_query(query)

        candidates: list[RetrievalResult] = []

        async with tenant_tx(ctx) as conn:
            all_cand_map: dict[uuid.UUID, RetrievalResult] = {}
            for q_iter in queries_to_search:
                iter_candidates: list[RetrievalResult] = []
                if request.method == "semantic":
                    assert query_vector is not None
                    iter_candidates = await search_semantic(
                        conn=conn,
                        query_vector=query_vector,
                        top_k=cand_k,
                        ctx=ctx,
                    )
                elif request.method == "keyword":
                    iter_candidates = await search_keyword(
                        conn=conn,
                        query_text=q_iter,
                        top_k=cand_k,
                        ctx=ctx,
                    )
                elif request.method == "hybrid":
                    iter_vec = query_vector if q_iter == query else await asyncio.to_thread(self._embedder.embed_query, q_iter)
                    iter_candidates = await search_hybrid(
                        conn=conn,
                        query_text=q_iter,
                        query_vector=iter_vec,
                        top_k=cand_k,
                        ctx=ctx,
                    )
                else:
                    raise ValueError(f"Unsupported retrieval method: {request.method}")

                for c in iter_candidates:
                    if c.chunk_id not in all_cand_map or c.score > all_cand_map[c.chunk_id].score:
                        all_cand_map[c.chunk_id] = c

            candidates = sorted(all_cand_map.values(), key=lambda x: x.score, reverse=True)[:cand_k]

            # Filter by optional similarity threshold if configured
            if settings.retrieval_similarity_threshold > 0:
                candidates = [r for r in candidates if r.score >= settings.retrieval_similarity_threshold]

            # Reranking stage: strictly operates on the authorized candidate set
            if should_rerank and candidates:
                try:
                    reranker = get_reranker()
                    candidates = reranker.rerank(query=query, candidates=candidates, top_k=cand_k)
                except Exception as e:
                    log.warning("Reranking failed: %s; falling back to hybrid candidate ranking", e)

            # Deduplication & Document Diversity optimization
            if settings.enable_deduplication and candidates:
                results = deduplicate_and_diversify(
                    candidates=candidates,
                    max_chunks_per_doc=settings.max_chunks_per_document,
                    similarity_threshold=settings.dedup_similarity_threshold,
                    top_k=top_k,
                )
            else:
                results = candidates[:top_k]


            # Audit logging: record retrieval attempt in the same transaction
            if record_audit:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                chunk_ids = [r.chunk_id for r in results]
                doc_ids = list(dict.fromkeys(r.document_id for r in results))
                query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

                await conn.execute(
                    text(
                        """
                        INSERT INTO retrieval_audit (
                            request_id, tenant_id, user_id, principals,
                            query_sha256, query_text, chunk_ids, document_ids, latency_ms
                        ) VALUES (
                            :request_id, CAST(:tenant_id AS uuid), :user_id, CAST(:principals AS text[]),
                            :query_sha256, :query_text, CAST(:chunk_ids AS uuid[]), CAST(:document_ids AS uuid[]), :latency_ms
                        )
                        """
                    ),
                    {
                        "request_id": str(req_id),
                        "tenant_id": ctx.tenant_id,
                        "user_id": ctx.user_id,
                        "principals": ctx.principals,
                        "query_sha256": query_hash,
                        "query_text": query[:500],
                        "chunk_ids": chunk_ids,
                        "document_ids": doc_ids,
                        "latency_ms": latency_ms,
                    },
                )

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
        )


retrieval_service = RetrievalService()
