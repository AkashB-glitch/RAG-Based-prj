import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local CPU embeddings (ONNX). bge-small-en-v1.5 -> 384 dims."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class FakeEmbedder:
    """Deterministic hashed bag-of-words. No downloads; good enough for tests and CI."""

    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder(settings.embedding_dim)
    if settings.embedder == "fastembed":
        try:
            return FastEmbedder(settings.embedding_model, settings.model_cache_dir)
        except (ImportError, ModuleNotFoundError):
            return FakeEmbedder(settings.embedding_dim)
    raise ValueError(f"unknown EMBEDDER={settings.embedder!r}")


def to_pgvector(v: list[float]) -> str:
    """pgvector text literal; cast with CAST(:x AS vector) in SQL."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
