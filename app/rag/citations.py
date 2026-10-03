"""Citation extraction and validation for PermRAG."""
from app.rag.schemas import SourceCitation
from app.retrieval.scoring import RetrievalResult


def build_citations(included_results: list[RetrievalResult]) -> list[SourceCitation]:
    """Builds deduplicated SourceCitation list preserving retrieval scores."""
    citations: list[SourceCitation] = []
    seen_chunks: set[str] = set()

    for r in included_results:
        if r.chunk_id not in seen_chunks:
            seen_chunks.add(r.chunk_id)
            citations.append(
                SourceCitation(
                    document_id=r.document_id,
                    document_name=r.document_name,
                    chunk_id=r.chunk_id,
                    score=r.score,
                )
            )

    return citations
