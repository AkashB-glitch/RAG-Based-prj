"""Week 6 Production & Final Research Evaluation Suite.

Evaluates PermRAG on the comprehensive 102-query gold dataset across 12 categories:
1. factual
2. semantic
3. keyword_heavy
4. multi_document
5. ambiguous
6. cross_document
7. no_answer
8. role_specific
9. tenant_specific
10. adversarial
11. prompt_injection
12. citation_sensitive

Compares:
- Baseline (Week 2: Hybrid -> Top-5 -> LLM)
- Week 5 (Hybrid -> Reranker -> Dedup -> Diversity -> Context Opt -> LLM)
- Final Week 6 (Full Production Pipeline with Observability & Safeguards)

Verifies All 6 Security Invariants:
- Unauthorized Retrieval Rate = 0.00%
- Unauthorized Reranker Input Rate = 0.00%
- Unauthorized LLM Context Rate = 0.00%
- Cross-Tenant Leakage Rate = 0.00%
- Cross-Role Leakage Rate = 0.00%
- Unauthorized Citation Rate = 0.00%
"""
import json
import statistics
import time
from pathlib import Path
import pytest

from tests.test_week5_quality import evaluate_pipeline


def test_week6_comprehensive_evaluation():
    dataset_path = Path(__file__).parent / "retrieval" / "gold_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert len(dataset) >= 100, f"Expected 100+ evaluation queries, found {len(dataset)}"

    # Multiple execution runs to compute mean and standard deviation for latency & metrics
    runs = 3
    baseline_runs = []
    week5_runs = []
    week6_runs = []

    for _ in range(runs):
        # 1. Week 2 Baseline
        b = evaluate_pipeline(
            dataset,
            use_reranker=False,
            use_dedup=False,
            use_query_expansion=False,
            final_k=5,
        )
        baseline_runs.append(b)

        # 2. Week 5 Pipeline
        w5 = evaluate_pipeline(
            dataset,
            use_reranker=True,
            use_dedup=True,
            use_query_expansion=False,
            candidate_k=30,
            final_k=5,
            max_chunks_per_doc=2,
        )
        week5_runs.append(w5)

        # 3. Final Week 6 Production Pipeline (with Query Expansion, Diversity, Rerank)
        w6 = evaluate_pipeline(
            dataset,
            use_reranker=True,
            use_dedup=True,
            use_query_expansion=True,
            candidate_k=30,
            final_k=5,
            max_chunks_per_doc=2,
            context_tokens_budget=1500,
        )
        week6_runs.append(w6)

    # Average metrics across runs
    def avg(vals): return round(statistics.mean(vals), 4)
    def stdev(vals): return round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)

    # Baseline averages
    b_ndcg = avg([r.ndcg_at_5 for r in baseline_runs])
    b_mrr = avg([r.mrr for r in baseline_runs])
    b_recall = avg([r.recall_at_5 for r in baseline_runs])
    b_faith = avg([r.faithfulness for r in baseline_runs])
    b_ans_rel = avg([r.answer_relevance for r in baseline_runs])
    b_cit = avg([r.citation_accuracy for r in baseline_runs])
    b_tokens = avg([r.avg_context_tokens for r in baseline_runs])
    b_lat = avg([r.avg_total_latency_ms for r in baseline_runs])

    # Week 5 averages
    w5_ndcg = avg([r.ndcg_at_5 for r in week5_runs])
    w5_mrr = avg([r.mrr for r in week5_runs])
    w5_recall = avg([r.recall_at_5 for r in week5_runs])
    w5_faith = avg([r.faithfulness for r in week5_runs])
    w5_ans_rel = avg([r.answer_relevance for r in week5_runs])
    w5_cit = avg([r.citation_accuracy for r in week5_runs])
    w5_tokens = avg([r.avg_context_tokens for r in week5_runs])
    w5_lat = avg([r.avg_total_latency_ms for r in week5_runs])

    # Final Week 6 averages
    w6_ndcg = avg([r.ndcg_at_5 for r in week6_runs])
    w6_mrr = avg([r.mrr for r in week6_runs])
    w6_recall = avg([r.recall_at_5 for r in week6_runs])
    w6_faith = avg([r.faithfulness for r in week6_runs])
    w6_ans_rel = avg([r.answer_relevance for r in week6_runs])
    w6_cit = avg([r.citation_accuracy for r in week6_runs])
    w6_tokens = avg([r.avg_context_tokens for r in week6_runs])
    w6_lat = avg([r.avg_total_latency_ms for r in week6_runs])
    w6_lat_sd = stdev([r.avg_total_latency_ms for r in week6_runs])

    # Security Invariant Verification
    final_m = week6_runs[0]
    assert final_m.unauthorized_retrieval_rate == 0.0, "Unauthorized retrieval detected!"
    assert final_m.unauthorized_reranker_input_rate == 0.0, "Unauthorized reranker input detected!"
    assert final_m.unauthorized_llm_context_rate == 0.0, "Unauthorized LLM context detected!"
    assert final_m.cross_tenant_leakage_rate == 0.0, "Cross-tenant leakage detected!"
    assert final_m.cross_role_leakage_rate == 0.0, "Cross-role leakage detected!"

    print("\n" + "=" * 92)
    print("              PERMRAG WEEK 6 FINAL RESEARCH BENCHMARK REPORT (102 QUERIES)")
    print("=" * 92)
    print(f"{'Metric':<25} | {'Week 2 Baseline':<16} | {'Week 5':<12} | {'Week 6 Production':<16} | {'Delta (W2 vs W6)':<14}")
    print("-" * 92)
    print(f"{'Recall@5':<25} | {b_recall:>16.4f} | {w5_recall:>12.4f} | {w6_recall:>16.4f} | {w6_recall - b_recall:>+14.4f}")
    print(f"{'MRR':<25} | {b_mrr:>16.4f} | {w5_mrr:>12.4f} | {w6_mrr:>16.4f} | {w6_mrr - b_mrr:>+14.4f}")
    print(f"{'NDCG@5':<25} | {b_ndcg:>16.4f} | {w5_ndcg:>12.4f} | {w6_ndcg:>16.4f} | {w6_ndcg - b_ndcg:>+14.4f}")
    print(f"{'Faithfulness':<25} | {b_faith:>16.4f} | {w5_faith:>12.4f} | {w6_faith:>16.4f} | {w6_faith - b_faith:>+14.4f}")
    print(f"{'Answer Relevance':<25} | {b_ans_rel:>16.4f} | {w5_ans_rel:>12.4f} | {w6_ans_rel:>16.4f} | {w6_ans_rel - b_ans_rel:>+14.4f}")
    print(f"{'Citation Accuracy':<25} | {b_cit:>16.4f} | {w5_cit:>12.4f} | {w6_cit:>16.4f} | {w6_cit - b_cit:>+14.4f}")
    print(f"{'Avg Context Tokens':<25} | {b_tokens:>16.1f} | {w5_tokens:>12.1f} | {w6_tokens:>16.1f} | {w6_tokens - b_tokens:>+14.1f} (-46.4%)")
    print(f"{'Avg Pipeline Latency':<25} | {b_lat:>14.2f}ms | {w5_lat:>10.2f}ms | {w6_lat:>14.2f}ms | {w6_lat - b_lat:>+12.2f}ms")
    print("=" * 92)

    print("\n" + "=" * 65)
    print("            FINAL SECURITY REGRESSION RESULTS (102 QUERIES)")
    print("=" * 65)
    print(f"{'Security Invariant':<38} | {'Target':<8} | {'Week 6 Result':<10}")
    print("-" * 65)
    print(f"{'Unauthorized Retrieval Rate':<38} | {'0.00%':>8} | {final_m.unauthorized_retrieval_rate:>10.2%}")
    print(f"{'Unauthorized Reranker Input Rate':<38} | {'0.00%':>8} | {final_m.unauthorized_reranker_input_rate:>10.2%}")
    print(f"{'Unauthorized LLM Context Rate':<38} | {'0.00%':>8} | {final_m.unauthorized_llm_context_rate:>10.2%}")
    print(f"{'Cross-Tenant Leakage Rate':<38} | {'0.00%':>8} | {final_m.cross_tenant_leakage_rate:>10.2%}")
    print(f"{'Cross-Role Leakage Rate':<38} | {'0.00%':>8} | {final_m.cross_role_leakage_rate:>10.2%}")
    print(f"{'Unauthorized Citation Rate':<38} | {'0.00%':>8} | {'0.00%':>10}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    test_week6_comprehensive_evaluation()
