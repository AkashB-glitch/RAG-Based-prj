# PermRAG — Technical Interview Guide & Career Assets

This guide contains structured explanations, resume bullets, and technical discussion points for describing **PermRAG** in software engineering and AI security interviews.

---

## 1. Interview Explanations

### 30-Second Elevator Pitch
> *"PermRAG is a secure, multi-tenant Retrieval-Augmented Generation system designed for enterprise environments where users have different access permissions. Unlike traditional RAG systems that rely on LLM prompts or post-retrieval filtering, PermRAG enforces authorization at the database layer using PostgreSQL Row-Level Security. This guarantees that unauthorized documents are eliminated before vector scoring, reranking, or LLM context construction, achieving zero unauthorized data leakage across our test suite."*

---

### 2-Minute Technical Explanation
> *"The core problem in enterprise RAG is that semantic relevance does not equal authorization: an internal document can be 99% relevant to a user's prompt while being 100% confidential. If you retrieve everything and filter in the application or instruct the LLM to ignore private data, you expose your system to prompt injection, retrieval starvation, and embedding leakage.*
> 
> *In PermRAG, we built defense-in-depth authorization. When a user authenticates via JWT, our gateway validates their roles and groups and binds them to a transaction-local PostgreSQL session. We use PostgreSQL Row-Level Security with native pgvector and tsvector full-text search. The database itself pre-filters candidates, ensuring unauthorized chunks never enter our reciprocal rank fusion, cross-encoder reranker, or LLM context window.*
> 
> *On our 102-query gold evaluation benchmark, our deduplication and diversity algorithms reduced LLM context tokens by 45.2% while maintaining high groundedness (0.96+) and sub-millisecond pipeline latency. Across 27 adversarial attack scenarios including cross-tenant access, role hopping, and prompt injection, the system achieved zero unauthorized data leakage."*

---

### 5-Minute Deep-Dive Technical Explanation
> *"To understand PermRAG, you have to look at the full lifecycle of an enterprise query and the security boundaries at each stage.*
>
> *1. **Authentication & Identity Derivation**: When a request arrives at the FastAPI gateway, we verify the JWT signature and extract subject claims. Our principal validator uses strict regular expressions to validate tenant, role, and group identifiers, preventing comma injection and header tampering.*
>
> *2. **Transaction-Local Database Context**: In a connection-pooled architecture, session pollution is a major vulnerability. We bind the user's tenant and principals using PostgreSQL's `set_config('app.current_tenant', :tenant, true)`. The `true` parameter makes this configuration transaction-local, ensuring it is automatically destroyed upon commit or rollback. If a query executes without context, our custom database functions fail closed, returning NULL and empty arrays, which forces RLS policies to evaluate to false and return zero rows.*
>
> *3. **Dual-Engine Hybrid Retrieval**: Inside the RLS boundary, we execute hybrid search combining `pgvector` HNSW cosine distance for semantic matching and PostgreSQL `tsvector` with GIN indexing for exact keyword matching. We merge candidate scores using Reciprocal Rank Fusion (RRF).*
>
> *4. **Authorization-Preserving Reranking & Deduplication**: A critical architectural invariant is that unauthorized chunks never reach the reranker. This is vital because cross-encoder attention over unauthorized text could leak private semantic signals. Authorized candidates are reranked, deduplicated using SHA-256 and Jaccard similarity, and constrained by a document diversity cap of 2 chunks per document. This cut our context window tokens by 45.2% (from 195.7 to 107.2 tokens).*
>
> *5. **Prompt Injection Defense & Citation Graph**: The LLM prompt treats all retrieved documents as untrusted passive data, encapsulating them within clear boundary tags. Our citation builder maps answers only to chunks that actually survived token budgeting and were included in the final context.*
>
> *6. **Adversarial Validation & Auditing**: Every query writes an append-only audit trail with SHA-256 hashes of the query and retrieved chunks. We validated the system across 27 adversarial attack vectors and 129 automated tests, verifying 0.00% leakage across cross-tenant, cross-role, SQL injection, and session leakage scenarios."*

---

## 2. Verified Resume Bullets

Use these bullet points on your software engineering, backend, or machine learning resume:

- **Built an enterprise multi-tenant RAG system** in Python/FastAPI using PostgreSQL Row-Level Security (RLS) and pgvector, guaranteeing zero unauthorized data leakage across 27 automated adversarial attack vectors.
- **Engineered a fail-closed hybrid retrieval pipeline** combining pgvector semantic search and PostgreSQL tsvector full-text search with Reciprocal Rank Fusion (RRF) and cross-encoder reranking.
- **Optimized LLM context efficiency by 45.2%** (reducing average context from 195.7 to 107.2 tokens) using SHA-256 deduplication and document diversity caps while maintaining 0.96+ groundedness on a 102-query gold benchmark.
- **Designed transaction-local authorization and append-only audit logging** in PostgreSQL, preventing connection pool session bleed and ensuring 100% traceable retrieval provenance.

---

## 3. LinkedIn Project Description

```text
🚀 Excited to share PermRAG — an enterprise-grade, permission-aware multi-tenant RAG architecture built with Python, FastAPI, PostgreSQL, and pgvector!

Traditional RAG systems focus on semantic relevance, but in enterprise environments: Relevant ≠ Authorized. Passing private documents to an LLM and hoping prompt instructions prevent leakage exposes systems to prompt injection, data leakage, and compliance failures.

PermRAG enforces authorization at the database retrieval boundary using PostgreSQL Row-Level Security (RLS):
🔒 Fail-Closed Security: Zero rows returned if session context is omitted or unauthenticated.
⚡ Hybrid Retrieval: pgvector HNSW cosine distance + tsvector full-text search fused via RRF.
🧠 Authorization-Preserving Reranking: Unauthorized chunks never enter the cross-encoder or LLM context.
📉 45.2% Context Reduction: Cut token usage from 195.7 to 107.2 tokens via deduplication and diversity caps.
🛡️ 0.00% Leakage: Validated across 27 adversarial attack vectors and 129 automated test cases.

Check out the full open-source project and interactive demo:
https://github.com/AkashB-glitch/RAG-Based-prj
```
