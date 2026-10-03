import re
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.principals import AuthContext
from app.retrieval.scoring import RetrievalResult


async def search_keyword(
    conn: AsyncConnection,
    query_text: str,
    top_k: int = 5,
    ctx: AuthContext | None = None,
    enforce_app_filter: bool = True,
) -> list[RetrievalResult]:
    """Execute PostgreSQL full-text search over chunks using websearch_to_tsquery and ts_rank.

    Dual-Layer Security:
    - Layer 1: Application-level WHERE clause filtering by tenant_id and acl && principals.
    - Layer 2: PostgreSQL Row-Level Security (RLS) prevents unauthorized chunks from matching,
      even without application-level filters.
    """
    clean_query = query_text.strip()
    # If query contains no alphanumeric characters, tsquery will be empty
    if not re.search(r"\w+", clean_query):
        return []

    params: dict = {
        "query": clean_query,
        "top_k": top_k,
    }

    where_clauses: list[str] = [
        "c.tsv @@ websearch_to_tsquery('english', :query)"
    ]

    if enforce_app_filter and ctx is not None:
        where_clauses.append("c.tenant_id = CAST(:tenant_id AS uuid)")
        where_clauses.append("c.acl && CAST(:principals AS text[])")
        params["tenant_id"] = ctx.tenant_id
        params["principals"] = ctx.principals

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = text(
        f"""
        SELECT c.id::text AS chunk_id,
               c.document_id::text AS document_id,
               d.title AS document_name,
               c.content,
               ts_rank(c.tsv, websearch_to_tsquery('english', :query)) AS score
          FROM chunks c
          JOIN documents d ON c.document_id = d.id
          {where_sql}
         ORDER BY score DESC
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
            score=round(float(r["score"]), 4),
            retrieval_method="keyword",
        )
        for r in rows
    ]
