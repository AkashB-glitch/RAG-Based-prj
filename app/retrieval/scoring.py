from typing import Literal
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    retrieval_method: Literal["semantic", "keyword", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query string")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Maximum number of chunks to return")
    method: Literal["hybrid", "semantic", "keyword"] = Field(default="hybrid", description="Retrieval method")
    rerank: bool | None = Field(default=None, description="Enable reranking stage")
    candidate_k: int | None = Field(default=None, ge=1, le=100, description="Number of candidate chunks before reranking")



class SearchResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    total_count: int


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize arbitrary non-negative scores into [0.0, 1.0] range."""
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0.0:
        return [0.0] * len(scores)
    return [max(0.0, min(1.0, s / max_score)) for s in scores]


def fuse_results(
    semantic_results: list[RetrievalResult],
    keyword_results: list[RetrievalResult],
    top_k: int,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_rrf: bool = False,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse semantic and keyword retrieval results using either normalized weighted scoring or RRF.

    Security critical: Does NOT add or manufacture results. Only combines items returned
    by the underlying permission-aware queries.
    """
    if not semantic_results and not keyword_results:
        return []

    # Map chunk_id -> {result_metadata, sem_score, kw_score, sem_rank, kw_rank}
    combined: dict[str, dict] = {}

    for rank, res in enumerate(semantic_results, start=1):
        if res.chunk_id not in combined:
            combined[res.chunk_id] = {
                "chunk_id": res.chunk_id,
                "document_id": res.document_id,
                "document_name": res.document_name,
                "content": res.content,
                "semantic_score": res.score,
                "keyword_score": 0.0,
                "semantic_rank": rank,
                "keyword_rank": None,
            }
        else:
            combined[res.chunk_id]["semantic_score"] = res.score
            combined[res.chunk_id]["semantic_rank"] = rank

    for rank, res in enumerate(keyword_results, start=1):
        if res.chunk_id not in combined:
            combined[res.chunk_id] = {
                "chunk_id": res.chunk_id,
                "document_id": res.document_id,
                "document_name": res.document_name,
                "content": res.content,
                "semantic_score": 0.0,
                "keyword_score": res.score,
                "semantic_rank": None,
                "keyword_rank": rank,
            }
        else:
            combined[res.chunk_id]["keyword_score"] = res.score
            combined[res.chunk_id]["keyword_rank"] = rank

    items = list(combined.values())

    if use_rrf:
        # Reciprocal Rank Fusion
        for item in items:
            rrf_sem = (semantic_weight / (rrf_k + item["semantic_rank"])) if item["semantic_rank"] is not None else 0.0
            rrf_kw = (keyword_weight / (rrf_k + item["keyword_rank"])) if item["keyword_rank"] is not None else 0.0
            item["fused_score"] = rrf_sem + rrf_kw
    else:
        # Normalized weighted linear combination
        raw_sem = [item["semantic_score"] for item in items]
        raw_kw = [item["keyword_score"] for item in items]
        norm_sem = normalize_scores(raw_sem)
        norm_kw = normalize_scores(raw_kw)

        total_weight = semantic_weight + keyword_weight or 1.0
        w_sem = semantic_weight / total_weight
        w_kw = keyword_weight / total_weight

        for i, item in enumerate(items):
            item["fused_score"] = (w_sem * norm_sem[i]) + (w_kw * norm_kw[i])

    items.sort(key=lambda x: x["fused_score"], reverse=True)

    return [
        RetrievalResult(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_name=item["document_name"],
            content=item["content"],
            score=round(float(item["fused_score"]), 4),
            retrieval_method="hybrid",
        )
        for item in items[:top_k]
    ]
