"""Quality and security evaluation test running against deterministic queries.json dataset.

Measures:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Unauthorized Retrieval Rate (URR) -> Target: 0.0%
- Cross-tenant leakage count -> Target: 0
- Cross-role leakage count -> Target: 0
- Missing-context result count -> Target: 0
"""
import json
from pathlib import Path
import pytest
import psycopg

from app.config import settings
from app.ingestion.embeddings import FakeEmbedder, to_pgvector
from tests.conftest import APP_DSN, requires_db
from tests.test_retrieval_security import retrieval_world

pytestmark = requires_db


def test_retrieval_quality_and_security_metrics(retrieval_world):
    queries_path = Path(__file__).parent / "retrieval" / "queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    emb = FakeEmbedder(settings.embedding_dim)
    top_k = 5

    total_queries = len(dataset)
    recall_hits = 0
    total_retrieved = 0
    relevant_retrieved = 0
    reciprocal_ranks = []

    # Security metrics
    cross_tenant_leakage = 0
    cross_role_leakage = 0
    unauthorized_chunks = 0
    missing_context_results = 0

    for item in dataset:
        q_text = item["query"]
        t_key = item["tenant"]
        role = item["role"]
        tenant_id = retrieval_world["tenants"][t_key]
        principals = [f"role:{role}", f"user:eval-{role}"]

        q_vec = to_pgvector(emb.embed_query(q_text))

        # Run retrieval under user's authenticated context
        with psycopg.connect(APP_DSN) as conn:
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            conn.execute("SELECT set_config('app.principals', %s, true)", (",".join(principals),))

            rows = conn.execute(
                """
                SELECT c.content, c.tenant_id::text, d.title
                  FROM chunks c
                  JOIN documents d ON c.document_id = d.id
                 ORDER BY c.embedding <=> %s::vector ASC
                 LIMIT %s
                """,
                (q_vec, top_k),
            ).fetchall()

        retrieved_contents = [r[0] for r in rows]
        retrieved_tenants = [r[1] for r in rows]
        total_retrieved += len(rows)

        # 1. Security check: Zero cross-tenant leakage
        for row_tenant in retrieved_tenants:
            if row_tenant != tenant_id:
                cross_tenant_leakage += 1
                unauthorized_chunks += 1

        # 2. Security check: Zero cross-role leakage
        for content in retrieved_contents:
            if role == "engineer" and ("salary" in content or "budget" in content or "finance" in content):
                cross_role_leakage += 1
                unauthorized_chunks += 1
            elif role == "hr" and ("deployment" in content or "budget" in content or "finance" in content):
                cross_role_leakage += 1
                unauthorized_chunks += 1
            elif role == "finance" and ("deployment" in content or "salary" in content):
                cross_role_leakage += 1
                unauthorized_chunks += 1

        # 3. Quality evaluation: Recall, Precision, MRR
        # Check if authorized expected document/chunk was matched
        matched_rank = None
        for rank, content in enumerate(retrieved_contents, start=1):
            if role in content.lower():
                relevant_retrieved += 1
                if matched_rank is None:
                    matched_rank = rank

        if matched_rank is not None:
            recall_hits += 1
            reciprocal_ranks.append(1.0 / matched_rank)
        else:
            reciprocal_ranks.append(0.0)

    # 4. Security check: Missing context test
    with psycopg.connect(APP_DSN) as conn:
        for item in dataset:
            q_vec = to_pgvector(emb.embed_query(item["query"]))
            # No context set: app.tenant_id and app.principals are empty
            rows = conn.execute(
                "SELECT c.id FROM chunks c ORDER BY c.embedding <=> %s::vector ASC LIMIT %s",
                (q_vec, top_k),
            ).fetchall()
            missing_context_results += len(rows)

    # Calculate metrics
    recall_at_k = recall_hits / total_queries if total_queries else 0.0
    precision_at_k = relevant_retrieved / total_retrieved if total_retrieved else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    unauthorized_retrieval_rate = unauthorized_chunks / (total_queries * top_k)

    print("\n" + "=" * 60)
    print("           WEEK 2 RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Evaluated Queries:       {total_queries}")
    print(f"Recall@{top_k}:                      {recall_at_k:.2%}")
    print(f"Precision@{top_k}:                   {precision_at_k:.2%}")
    print(f"Mean Reciprocal Rank (MRR):    {mrr:.4f}")
    print("-" * 60)
    print(f"Unauthorized Retrieval Rate:   {unauthorized_retrieval_rate:.2%} (Target: 0.0%)")
    print(f"Cross-Tenant Leakage Count:    {cross_tenant_leakage}")
    print(f"Cross-Role Leakage Count:      {cross_role_leakage}")
    print(f"Missing-Context Leakage:       {missing_context_results}")
    print("=" * 60)

    # Strict security assertions
    assert unauthorized_retrieval_rate == 0.0, f"Unauthorized retrieval rate was {unauthorized_retrieval_rate}!"
    assert cross_tenant_leakage == 0, f"Cross-tenant leaks detected: {cross_tenant_leakage}"
    assert cross_role_leakage == 0, f"Cross-role leaks detected: {cross_role_leakage}"
    assert missing_context_results == 0, f"Missing-context leaked rows: {missing_context_results}"
    assert recall_at_k > 0.0, "Recall should be positive for authorized queries"
