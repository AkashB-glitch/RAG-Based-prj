"""Deduplication and document-level diversity optimization for PermRAG."""
import hashlib
import re

from app.config import settings
from app.retrieval.scoring import RetrievalResult


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    set_a = _token_set(text_a)
    set_b = _token_set(text_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_and_diversify(
    candidates: list[RetrievalResult],
    max_chunks_per_doc: int | None = None,
    similarity_threshold: float | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Deduplicates near-identical chunks and enforces document diversity.

    - Removes exact text duplicates using SHA-256 hashes.
    - Removes near-duplicates with Jaccard token overlap > similarity_threshold.
    - Enforces max_chunks_per_doc to prevent one document from monopolizing context.
    """
    if not candidates:
        return []

    max_per_doc = max_chunks_per_doc or settings.max_chunks_per_document
    threshold = similarity_threshold or settings.dedup_similarity_threshold
    limit = top_k or len(candidates)

    accepted: list[RetrievalResult] = []
    seen_hashes: set[str] = set()
    doc_chunk_counts: dict[str, int] = {}

    for c in candidates:
        # 1. Exact hash deduplication
        norm_text = " ".join(c.content.lower().split())
        h = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue

        # 2. Document diversity cap
        doc_count = doc_chunk_counts.get(c.document_id, 0)
        if doc_count >= max_per_doc:
            continue

        # 3. Near-duplicate suppression against already accepted chunks
        is_near_duplicate = False
        for acc in accepted:
            if jaccard_similarity(c.content, acc.content) >= threshold:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        # Accept chunk
        seen_hashes.add(h)
        doc_chunk_counts[c.document_id] = doc_count + 1
        accepted.append(c)

        if len(accepted) >= limit:
            break

    return accepted
