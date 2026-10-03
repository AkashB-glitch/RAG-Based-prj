from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    score: float


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Max chunks to retrieve")
    candidate_k: int | None = Field(default=None, ge=1, le=200, description="Max candidates before reranking")
    rerank: bool | None = Field(default=None, description="Whether to rerank candidates")


class LatencyBreakdown(BaseModel):
    auth_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    context_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    latency: LatencyBreakdown | None = None
