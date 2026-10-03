"""Lightweight property-based and fuzz testing for query, principal, and metadata handling."""
import random
import string
import pytest

from app.principals import PRINCIPAL_RE, validate_principal
from app.rag.context import ContextBuilder
from app.retrieval.scoring import RetrievalResult, normalize_scores


FUZZ_CHARSET = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?/`~ \t\n\r" + "你好世界🚀éèñ"


def generate_fuzz_string(min_len: int = 1, max_len: int = 200) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(FUZZ_CHARSET) for _ in range(length))


def test_fuzz_principal_validation():
    """Fuzz testing principal validation: only valid user|role|group formatted strings can pass."""
    for _ in range(200):
        fuzz_str = generate_fuzz_string(1, 100)
        # Only strings strictly conforming to regex must succeed
        if PRINCIPAL_RE.fullmatch(fuzz_str):
            assert validate_principal(fuzz_str) == fuzz_str
        else:
            with pytest.raises(ValueError):
                validate_principal(fuzz_str)


def test_fuzz_score_normalization():
    """Fuzz testing normalize_scores with random, negative, extreme values."""
    for _ in range(100):
        count = random.randint(0, 50)
        scores = [random.uniform(-1000.0, 1000.0) for _ in range(count)]
        norm = normalize_scores(scores)
        assert len(norm) == len(scores)
        for s in norm:
            assert 0.0 <= s <= 1.0


def test_fuzz_context_builder_invariants():
    """Fuzz testing ContextBuilder with random chunks, unicode, and varied budgets."""
    for _ in range(50):
        max_chunks = random.randint(1, 10)
        max_tokens = random.randint(10, 500)
        max_chunk_length = random.randint(10, 100)

        cb = ContextBuilder(
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            max_chunk_length=max_chunk_length,
        )

        num_results = random.randint(0, 15)
        mock_results = [
            RetrievalResult(
                chunk_id=f"c-{i}",
                document_id=f"d-{i}",
                document_name=f"doc_{generate_fuzz_string(3, 10)}.pdf",
                content=generate_fuzz_string(5, 300),
                score=round(random.random(), 4),
                retrieval_method="hybrid",
            )
            for i in range(num_results)
        ]

        ctx_str, included = cb.build_context(mock_results)

        # Invariant 1: AUTHORIZED_CONTEXT ⊆ AUTHORIZED_RETRIEVAL_RESULTS
        assert len(included) <= max_chunks
        assert set(r.chunk_id for r in included) <= set(r.chunk_id for r in mock_results)
        for r in included:
            assert r.chunk_id in ctx_str
