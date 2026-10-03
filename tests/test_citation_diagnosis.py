"""Diagnostic experiment for Week 6: Investigate Citation Accuracy delta between Week 2 and Week 5.

Week 2 Citation Accuracy: 0.8722
Week 5 Citation Accuracy: 0.8611

Isolates:
1. Reranker impact
2. Deduplication impact
3. Document diversity cap impact (MAX_CHUNKS_PER_DOCUMENT = 2 vs unconstrained)
4. Query expansion impact
"""
import json
from pathlib import Path

from tests.test_week5_quality import evaluate_pipeline


def run_citation_diagnosis():
    dataset_path = Path(__file__).parent / "retrieval" / "gold_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 1. Baseline: No reranker, no dedup, no diversity cap
    m_base = evaluate_pipeline(dataset, use_reranker=False, use_dedup=False, use_query_expansion=False, final_k=5)

    # 2. Add Reranker only
    m_rerank = evaluate_pipeline(dataset, use_reranker=True, use_dedup=False, use_query_expansion=False, candidate_k=30, final_k=5)

    # 3. Add Deduplication only (unconstrained diversity cap: max 10 per doc)
    m_dedup_only = evaluate_pipeline(dataset, use_reranker=True, use_dedup=True, use_query_expansion=False, candidate_k=30, final_k=5, max_chunks_per_doc=10)

    # 4. Add Document Diversity Cap (max 2 per doc)
    m_diversity_cap = evaluate_pipeline(dataset, use_reranker=True, use_dedup=True, use_query_expansion=False, candidate_k=30, final_k=5, max_chunks_per_doc=2)

    # 5. Full Pipeline with Query Expansion
    m_full = evaluate_pipeline(dataset, use_reranker=True, use_dedup=True, use_query_expansion=True, candidate_k=30, final_k=5, max_chunks_per_doc=2)

    print("\n" + "=" * 80)
    print("         CITATION ACCURACY DIAGNOSTIC EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Pipeline Variant':<35} | {'Citation Acc':<12} | {'NDCG@5':<8} | {'Avg Tokens':<10}")
    print("-" * 80)
    print(f"{'1. Baseline (Week 2)':<35} | {m_base.citation_accuracy:>12.4f} | {m_base.ndcg_at_5:>8.4f} | {m_base.avg_context_tokens:>10.1f}")
    print(f"{'2. + Reranker Only':<35} | {m_rerank.citation_accuracy:>12.4f} | {m_rerank.ndcg_at_5:>8.4f} | {m_rerank.avg_context_tokens:>10.1f}")
    print(f"{'3. + Dedup Only (no diversity cap)':<35} | {m_dedup_only.citation_accuracy:>12.4f} | {m_dedup_only.ndcg_at_5:>8.4f} | {m_dedup_only.avg_context_tokens:>10.1f}")
    print(f"{'4. + Diversity Cap (max 2 / doc)':<35} | {m_diversity_cap.citation_accuracy:>12.4f} | {m_diversity_cap.ndcg_at_5:>8.4f} | {m_diversity_cap.avg_context_tokens:>10.1f}")
    print(f"{'5. + Query Expansion (Full Week 5/6)':<35} | {m_full.citation_accuracy:>12.4f} | {m_full.ndcg_at_5:>8.4f} | {m_full.avg_context_tokens:>10.1f}")
    print("=" * 80 + "\n")


def test_citation_diagnosis():
    run_citation_diagnosis()


if __name__ == "__main__":
    run_citation_diagnosis()
