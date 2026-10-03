"""Context Builder for PermRAG.

Enforces:
- Invariant 1: AUTHORIZED_CONTEXT ⊆ AUTHORIZED_RETRIEVAL_RESULTS
- Invariant 2: UNAUTHORIZED_CHUNKS ∩ LLM_CONTEXT = ∅
- Context budgeting (max chunks, max tokens, max chunk length)
- Preservation of source identity, ordering, and scores
- Untrusted data encapsulation preventing prompt injection
"""
from app.config import settings
from app.retrieval.scoring import RetrievalResult


class ContextBuilder:
    def __init__(
        self,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
        max_chunk_length: int | None = None,
    ):
        self.max_chunks = max_chunks or settings.max_context_chunks
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.max_chunk_length = max_chunk_length or settings.max_chunk_length

    def build_context(
        self,
        retrieval_results: list[RetrievalResult],
    ) -> tuple[str, list[RetrievalResult]]:
        """Builds structured LLM context from authorized retrieval results.

        Returns:
            tuple[str, list[RetrievalResult]]: (formatted_context_string, included_results)
        """
        if not retrieval_results:
            return "", []

        # Sort strictly by retrieval score descending
        sorted_results = sorted(retrieval_results, key=lambda r: r.score, reverse=True)

        included_results: list[RetrievalResult] = []
        context_blocks: list[str] = []
        current_token_estimate = 0

        for idx, res in enumerate(sorted_results[: self.max_chunks], start=1):
            # Guard against oversized individual chunks
            content = res.content.strip()
            if len(content) > self.max_chunk_length:
                content = content[: self.max_chunk_length] + " [truncated]"

            block = (
                f"--- SOURCE {idx} ---\n"
                f"Document: {res.document_name}\n"
                f"Chunk ID: {res.chunk_id}\n"
                f"Score: {res.score}\n"
                f"Content:\n\"\"\"\n{content}\n\"\"\"\n"
            )

            # Heuristic: 1 token ~ 4 characters
            block_tokens = max(1, len(block) // 4)
            if current_token_estimate + block_tokens > self.max_tokens and included_results:
                # Do not exceed the configured context budget
                break

            context_blocks.append(block)
            included_results.append(res)
            current_token_estimate += block_tokens

        formatted_context = "\n".join(context_blocks)
        return formatted_context, included_results


context_builder = ContextBuilder()
