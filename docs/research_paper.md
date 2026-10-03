# PermRAG: Fail-Closed Permission-Aware Hybrid Retrieval for Multi-Tenant Enterprise Generative Systems

## Formal Abstract

Enterprise adoption of Retrieval-Augmented Generation (RAG) is fundamentally constrained by data isolation and access control requirements. Conventional RAG architectures optimize semantic similarity and generative fluency while treating authorization as an afterthought, either relying on post-retrieval application filters or prompt-based instruction following. This paper presents **PermRAG**, a permission-aware multi-tenant RAG architecture that enforces access authorization at the database retrieval boundary. By leveraging transaction-local PostgreSQL Row-Level Security (RLS) in conjunction with pgvector and full-text search, PermRAG guarantees that unauthorized chunks are eliminated prior to candidate fusion, cross-encoder reranking, and context construction. We formalize four operational security invariants that prevent unauthorized information leakage across the entire pipeline. In an empirical evaluation spanning a 102-query gold evaluation dataset across multi-tenant enterprise domains, PermRAG achieved 0.00% unauthorized retrieval, 0.00% reranker input leakage, 0.00% context leakage, and 0.00% citation leakage across 27 adversarial attack vectors. Furthermore, our near-duplicate deduplication and document diversity capping reduced average context window tokens by 45.2% (195.7 to 107.2 tokens) while maintaining high factual groundedness (0.96+) and sub-millisecond pipeline latency.

---

## Research Paper Structure & Outline

### I. Introduction
- The Enterprise RAG Paradox: Balancing Information Access and Confidentiality.
- The Fallacy of Prompt-Level Security: Why "Don't Look at This" Fails.
- Limitations of Post-Retrieval Filtering: Retrieval Starvation and Side-Channel Leakage.
- Core Contribution: Pre-retrieval database-enforced authorization and invariant preservation.

### II. Related Work
- Retrieval-Augmented Generation (Lewis et al., Guu et al.).
- Multi-Tenant Database Security and Row-Level Security (RLS) Models.
- Prompt Injection and Indirect Adversarial Attacks on LLMs.
- Hybrid Search and Candidate Score Fusion (Reciprocal Rank Fusion).

### III. Problem Definition & Formal Invariants
- Mathematical Definition of Multi-Tenant Access Control: Tenants, Roles, Groups, and ACLs.
- The Core Premise: $\text{Relevant} \neq \text{Authorized}$.
- Formulation of the Four Security Invariants:
  1. $\text{RERANKER\_INPUT} \subseteq \text{AUTHORIZED\_RETRIEVAL\_RESULTS}$
  2. $\text{UNAUTHORIZED\_CHUNKS} \cap \text{RERANKER\_INPUT} = \emptyset$
  3. $\text{UNAUTHORIZED\_CHUNKS} \cap \text{LLM\_CONTEXT} = \emptyset$
  4. $\text{CITED\_SOURCES} \subseteq \text{FINAL\_LLM\_CONTEXT}$

### IV. System Architecture
- Gateway and Authentication Layer: Stateless JWT Verification and Principal Extraction.
- Transaction-Local Session Binding via PostgreSQL `set_config(..., is_local=true)`.
- Fail-Closed Security Primitives: `app_tenant()`, `app_principals()`, `app_user_id()`.
- Least-Privilege Database Role Topology (`app_user`, `rag_worker`, `permrag_owner`).

### V. Permission-Aware Hybrid Retrieval
- Dual-Engine Retrieval:
  - Dense Vector Semantic Search with pgvector (HNSW Cosine Distance).
  - Lexical Keyword Full-Text Search with PostgreSQL tsvector (GIN Indexing).
- Normalized Reciprocal Rank Fusion (RRF) inside the Authorization Boundary.
- Security Guarantees: Why Pre-Filtering Eliminates Candidate Set Contamination.

### VI. Secure RAG Generation & Context Optimization
- Cross-Encoder Reranking: Invariant Verification and Isolation.
- Content Optimization: SHA-256 Content Hashing and Jaccard Near-Duplicate Filtering.
- Document Diversity Constraints (`MAX_CHUNKS_PER_DOC = 2`).
- Token Budgeting and Prompt Encapsulation for Indirect Prompt Injection Neutralization.
- Grounded Citation Graph Construction.

### VII. Threat Model & Adversarial Analysis
- Identification of 9 Primary Threat Vectors (Cross-Tenant, Cross-Role, SQLi, Prompt Injection, Session Bleed, etc.).
- Formal Defense Mechanisms and Architectural Isolation Guarantees.

### VIII. Experimental Methodology
- Construction of the 102-Query Enterprise Gold Evaluation Dataset.
- Breakdown of 12 Query Categories: Factual, Semantic, Keyword, Multi-Doc, Adversarial, etc.
- Hardware, Execution Environment, and Automated Testing Setup.

### IX. Retrieval Quality & Efficiency Results
- Recall@5, MRR, and NDCG@5 Performance Metrics.
- Context Efficiency Analysis: 45.2% Token Reduction (195.7 $\rightarrow$ 107.2 tokens).
- Pipeline Latency Profiling across Retrieval, Reranking, Context, and LLM stages.

### X. Security Evaluation & Adversarial Attack Simulation
- Evaluation of 27 Structured Adversarial Test Scenarios.
- Empirical Leakage Rates: 0.00% across all evaluated security invariants.
- Failure Analysis of Application-Level Only Security vs. Database-Enforced Defense in Depth.

### XI. Citation Accuracy Diagnostic Investigation
- Analysis of Citation Accuracy Shift (0.8722 $\rightarrow$ 0.8611).
- Isolation of Reranking vs. Deduplication Effects: Proving Deduplication Eliminates Duplicate Target Artifacts.

### XII. Limitations & Future Work
- Evaluation Dataset Synthetic Bounds.
- Model Dependencies and Hardware Requirements.
- Extensions to Attribute-Based Access Control (ABAC) and Policy-as-Code (Open Policy Agent).
- Scalability Considerations for Multi-Million Document Repositories.

### XIII. Conclusion
- Summary of Findings and Architectural Recommendations for Secure Enterprise Generative Systems.
