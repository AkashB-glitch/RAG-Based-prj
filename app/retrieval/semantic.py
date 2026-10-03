from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.ingestion.embeddings import to_pgvector
from app.principals import AuthContext
from app.retrieval.scoring import RetrievalResult


async def search_semantic(
    conn: AsyncConnection,
    query_vector: list[float],
    top_k: int = 5,
    ctx: AuthContext | None = None,
    enforce_app_filter: bool = True,
) -> list[RetrievalResult]:
    """Execute vector similarity search over chunks.

    Dual-Layer Security:
    - Layer 1: Application-level WHERE clause filtering by tenant_id and acl && principals (if enforce_app_filter=True).
    - Layer 2: PostgreSQL Row-Level Security (RLS) enforces the same policy inside the database engine.
      Even if enforce_app_filter=False or developer omits WHERE, PostgreSQL RLS physically prevents
      unauthorized chunks from being returned or ranked.
    """
    vector_literal = to_pgvector(query_vector)

    params: dict = {
        "query_vector": vector_literal,
        "top_k": top_k,
    }

    where_clauses: list[str] = []
    if enforce_app_filter and ctx is not None:
        where_clauses.append("c.tenant_id = CAST(:tenant_id AS uuid)")
        where_clauses.append("c.acl && CAST(:principals AS text[])")
        params["tenant_id"] = ctx.tenant_id
        params["principals"] = ctx.principals

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = text(
        f"""
        SELECT c.id::text AS chunk_id,
               c.document_id::text AS document_id,
               d.title AS document_name,
               c.content,
               (1 - (c.embedding <=> CAST(:query_vector AS vector))) AS score
          FROM chunks c
          JOIN documents d ON c.document_id = d.id
          {where_sql}
         ORDER BY c.embedding <=> CAST(:query_vector AS vector) ASC
         LIMIT :top_k
        """
    )

    rows = (await conn.execute(query, params)).mappings().all()

    return [
        RetrievalResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            document_name=r["document_name"],
            content=r["content"],
            score=round(max(0.0, float(r["score"])), 4),
            retrieval_method="semantic",
        )
        for r in rows
    ]
