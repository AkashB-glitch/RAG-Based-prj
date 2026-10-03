from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"

    owner_database_url: str = "postgresql+psycopg://permrag_owner:owner_password@localhost:5432/permrag"
    app_database_url: str = "postgresql+asyncpg://app_user:app_password@localhost:5432/permrag"
    worker_database_url: str = "postgresql+asyncpg://rag_worker:worker_password@localhost:5432/permrag"

    jwt_secret: str = "change-me-to-a-long-random-string"
    jwt_issuer: str = "permrag-dev"
    jwt_audience: str = "permrag-api"
    jwt_ttl_minutes: int = 60

    embedder: str = "fastembed"  # "fastembed" | "fake"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    model_cache_dir: str = "./.models"

    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200

    worker_poll_seconds: float = 2.0
    worker_max_attempts: int = 5
    worker_stale_lock_minutes: int = 10

    # Retrieval settings
    retrieval_top_k: int = 5
    retrieval_semantic_weight: float = 0.7
    retrieval_keyword_weight: float = 0.3
    retrieval_rrf_k: int = 60
    retrieval_similarity_threshold: float = 0.0

    # RAG & LLM settings
    llm_provider: str = "mock"  # "mock" | "openai" | "generic"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 30.0

    max_context_chunks: int = 5
    max_context_tokens: int = 4000
    max_chunk_length: int = 1500

    # Week 5: Advanced Retrieval, Reranking & Deduplication
    reranker_provider: str = "mock"  # "mock" | "cross-encoder" | "fastembed"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 30
    final_k: int = 5
    max_chunks_per_document: int = 2
    dedup_similarity_threshold: float = 0.85
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    # Week 6: Productionization, Observability & Rate Limiting
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    rate_limit_auth_per_minute: int = 30
    rate_limit_upload_per_minute: int = 20
    rate_limit_rag_per_minute: int = 60
    allowed_upload_extensions: list[str] = [".txt", ".md", ".pdf"]
    cors_origins: list[str] = ["*"]


settings = Settings()
