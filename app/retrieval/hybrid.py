from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.principals import AuthContext
from app.retrieval.keyword import search_keyword
from app.retrieval.scoring import RetrievalResult, fuse_results
from app.retrieval.semantic import search_semantic


async def search_hybrid(
    conn: AsyncConnection,
    query_text: str,
    query_vector: list[float],
    top_k: int = 5,
    ctx: AuthContext | None = None,
    semantic_weight: float | None = None,
    keyword_weight: float | None = None,
    use_rrf: bool = False,
    rrf_k: int | None = None,
    enforce_app_filter: bool = True,
) -> list[RetrievalResult]:
    """Execute hybrid retrieval combining semantic vector search and PostgreSQL full-text search.

    Candidate results from both methods are fused without bypassing PostgreSQL RLS boundaries.
    """
    sem_w = semantic_weight if semantic_weight is not None else settings.retrieval_semantic_weight
    kw_w = keyword_weight if keyword_weight is not None else settings.retrieval_keyword_weight
    k_rrf = rrf_k if rrf_k is not None else settings.retrieval_rrf_k

    # Retrieve candidate pool from each strategy
    candidate_k = max(top_k * 2, 10)

    semantic_candidates = await search_semantic(
        conn=conn,
        query_vector=query_vector,
        top_k=candidate_k,
        ctx=ctx,
        enforce_app_filter=enforce_app_filter,
    )

    keyword_candidates = await search_keyword(
        conn=conn,
        query_text=query_text,
        top_k=candidate_k,
        ctx=ctx,
        enforce_app_filter=enforce_app_filter,
    )

    return fuse_results(
        semantic_results=semantic_candidates,
        keyword_results=keyword_candidates,
        top_k=top_k,
        semantic_weight=sem_w,
        keyword_weight=kw_w,
        use_rrf=use_rrf,
        rrf_k=k_rrf,
    )
