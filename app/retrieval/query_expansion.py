"""Query expansion and rewriting for PermRAG.

Security Invariant:
- Query expansion is strictly a retrieval candidate optimization.
- The original query is preserved for LLM generation, responses, and audit logging.
- Expanded queries NEVER modify or influence tenant, role, or authorization state.
"""
import re

SYNONYM_MAP: dict[str, list[str]] = {
    "deployment": ["release", "rollout", "procedure"],
    "deploying": ["release", "rollout", "procedure"],
    "deploy": ["release", "rollout", "procedure"],
    "rollback": ["revert", "recovery"],
    "salary": ["compensation", "pay", "merit"],
    "compensation": ["salary", "pay", "merit"],
    "vacation": ["leave", "holiday", "time off"],
    "leave": ["vacation", "holiday", "time off"],
    "budget": ["fiscal", "forecast", "revenue"],
    "financial": ["budget", "fiscal", "revenue"],
    "audit": ["compliance", "accounting"],
    "container": ["docker", "image"],
    "kubernetes": ["k8s", "cluster"],
    "medical": ["health", "insurance", "dental"],
    "insurance": ["medical", "health", "benefits"],
}


def expand_query(query: str, max_variations: int = 2) -> list[str]:
    """Expands user query with domain-specific search variations.

    Returns:
        list[str]: [original_query, variation1, ...]
    """
    clean = query.strip()
    if not clean:
        return [query]

    variations = [clean]
    words = re.findall(r"\w+", clean.lower())

    # Build alternate query by substituting recognized keywords with canonical synonyms
    alternate_tokens = []
    has_substitution = False

    for w in words:
        if w in SYNONYM_MAP and not has_substitution:
            alternate_tokens.append(SYNONYM_MAP[w][0])
            has_substitution = True
        else:
            alternate_tokens.append(w)

    if has_substitution:
        alt_query = " ".join(alternate_tokens)
        if alt_query.lower() != clean.lower():
            variations.append(alt_query)

    return variations[: max_variations + 1]
