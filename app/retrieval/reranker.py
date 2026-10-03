"""Reranking engine for PermRAG.

Critical Invariant:
- RERANKER_INPUT ⊆ AUTHORIZED_RETRIEVAL_RESULTS
- UNAUTHORIZED_CHUNKS ∩ RERANKER_INPUT = ∅
- UNAUTHORIZED_RERANKER_INPUT_RATE = 0%
"""
import logging
import re
from typing import Protocol

from app.config import settings
from app.retrieval.scoring import RetrievalResult

log = logging.getLogger("reranker")


class BaseReranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        ...


class MockReranker:
    """Mock Reranker for deterministic testing, CI, and security inspection.

    Captures last_inputs to prove that unauthorized chunks never entered the reranker.
    """

    def __init__(self):
        self.last_inputs: list[RetrievalResult] = []
        self.last_query: str | None = None
        self.rerank_count: int = 0

    def reset(self):
        self.last_inputs = []
        self.last_query = None
        self.rerank_count = 0

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        self.last_inputs = list(candidates)
        self.last_query = query
        self.rerank_count += 1

        if not candidates:
            return []

        query_terms = set(re.findall(r"\w{3,}", query.lower()))

        scored_candidates = []
        for c in candidates:
            content_terms = set(re.findall(r"\w{3,}", c.content.lower()))
            overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0

            # Fused rerank score: 50% candidate hybrid score + 50% lexical relevance
            rerank_score = round(0.5 * c.score + 0.5 * overlap, 4)

            scored_candidates.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_name=c.document_name,
                    content=c.content,
                    score=rerank_score,
                    retrieval_method=c.retrieval_method,
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]


class FastEmbedReranker:
    """Local CPU Cross-Encoder reranker using fastembed / onnxruntime if available."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.last_inputs: list[RetrievalResult] = []

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._model = TextCrossEncoder(model_name=self.model_name)
            except Exception as e:
                log.warning("Could not load TextCrossEncoder: %s; falling back to relevance scorer", e)
                return None
        return self._model

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        self.last_inputs = list(candidates)
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            # Fallback to deterministic scoring
            return _mock_reranker.rerank(query, candidates, top_k)

        try:
            texts = [c.content for c in candidates]
            scores = list(model.rerank(query, texts))

            scored = []
            for c, s in zip(candidates, scores):
                scored.append(
                    RetrievalResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        document_name=c.document_name,
                        content=c.content,
                        score=round(float(s), 4),
                        retrieval_method=c.retrieval_method,
                    )
                )
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:top_k]
        except Exception as e:
            log.warning("Reranking failed: %s; preserving candidate ordering", e)
            return candidates[:top_k]


_mock_reranker = MockReranker()


def get_reranker() -> BaseReranker:
    if settings.reranker_provider == "cross-encoder" or settings.reranker_provider == "fastembed":
        return FastEmbedReranker(settings.reranker_model)
    return _mock_reranker
