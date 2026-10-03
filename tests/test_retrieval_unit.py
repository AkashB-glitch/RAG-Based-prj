import pytest

from app.ingestion.embeddings import FakeEmbedder
from app.retrieval.scoring import (
    RetrievalResult,
    SearchRequest,
    SearchResponse,
    fuse_results,
    normalize_scores,
)


def test_retrieval_result_schema():
    res = RetrievalResult(
        chunk_id="c1",
        document_id="d1",
        document_name="spec.pdf",
        content="Deploying to cluster",
        score=0.88,
        retrieval_method="semantic",
    )
    assert res.chunk_id == "c1"
    assert res.retrieval_method == "semantic"
    assert res.score == 0.88


def test_search_request_validation():
    req = SearchRequest(query="how to deploy", top_k=5)
    assert req.query == "how to deploy"
    assert req.top_k == 5
    assert req.method == "hybrid"

    # Reject empty query
    with pytest.raises(ValueError):
        SearchRequest(query="")

    # Reject top_k < 1
    with pytest.raises(ValueError):
        SearchRequest(query="valid", top_k=0)


def test_normalize_scores():
    assert normalize_scores([]) == []
    assert normalize_scores([0.0, 0.0]) == [0.0, 0.0]

    norm = normalize_scores([2.0, 4.0, 10.0])
    assert norm == [0.2, 0.4, 1.0]


def test_fuse_results_weighted():
    sem = [
        RetrievalResult(chunk_id="c1", document_id="d1", document_name="doc1", content="t1", score=0.9, retrieval_method="semantic"),
        RetrievalResult(chunk_id="c2", document_id="d2", document_name="doc2", content="t2", score=0.6, retrieval_method="semantic"),
    ]
    kw = [
        RetrievalResult(chunk_id="c2", document_id="d2", document_name="doc2", content="t2", score=0.8, retrieval_method="keyword"),
        RetrievalResult(chunk_id="c3", document_id="d3", document_name="doc3", content="t3", score=0.5, retrieval_method="keyword"),
    ]

    fused = fuse_results(sem, kw, top_k=3, semantic_weight=0.7, keyword_weight=0.3, use_rrf=False)
    assert len(fused) == 3
    # c2 is in both lists; c1 is top in semantic; all have retrieval_method="hybrid"
    assert all(r.retrieval_method == "hybrid" for r in fused)
    chunk_ids = [r.chunk_id for r in fused]
    assert set(chunk_ids) == {"c1", "c2", "c3"}
    # Scores are sorted descending
    assert fused[0].score >= fused[1].score >= fused[2].score


def test_fuse_results_rrf():
    sem = [
        RetrievalResult(chunk_id="c1", document_id="d1", document_name="doc1", content="t1", score=0.95, retrieval_method="semantic"),
        RetrievalResult(chunk_id="c2", document_id="d2", document_name="doc2", content="t2", score=0.80, retrieval_method="semantic"),
    ]
    kw = [
        RetrievalResult(chunk_id="c2", document_id="d2", document_name="doc2", content="t2", score=0.90, retrieval_method="keyword"),
        RetrievalResult(chunk_id="c1", document_id="d1", document_name="doc1", content="t1", score=0.40, retrieval_method="keyword"),
    ]

    fused = fuse_results(sem, kw, top_k=2, semantic_weight=0.5, keyword_weight=0.5, use_rrf=True, rrf_k=60)
    assert len(fused) == 2
    assert all(r.retrieval_method == "hybrid" for r in fused)


def test_fuse_results_empty():
    assert fuse_results([], [], top_k=5) == []


def test_fake_embedder_query():
    emb = FakeEmbedder(dim=384)
    v = emb.embed_query("test query string")
    assert len(v) == 384
    # Length of normalized vector should be close to 1.0
    import math
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-4
