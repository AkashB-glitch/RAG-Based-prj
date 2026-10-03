"""LLM provider abstraction for PermRAG."""
import re
from typing import Protocol
import httpx

from app.config import settings
from app.rag.prompts import INSUFFICIENT_INFO_MESSAGE, NO_AUTHORIZED_INFO_MESSAGE


class LLMProvider(Protocol):
    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        ...


class MockLLMProvider:
    """Mock LLM Provider for deterministic testing and CI.

    Captures last_context, last_system_prompt, and last_user_query to allow tests
    to inspect and assert that unauthorized data NEVER reached the LLM context.
    """

    def __init__(self):
        self.last_context: str | None = None
        self.last_system_prompt: str | None = None
        self.last_user_query: str | None = None
        self.generation_count: int = 0

    def reset(self):
        self.last_context = None
        self.last_system_prompt = None
        self.last_user_query = None
        self.generation_count = 0

    def generate_sync(self, system_prompt: str, user_query: str, context: str) -> str:
        return self._generate_internal(system_prompt, user_query, context)

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        return self._generate_internal(system_prompt, user_query, context)

    def _generate_internal(self, system_prompt: str, user_query: str, context: str) -> str:
        self.last_context = context
        self.last_system_prompt = system_prompt
        self.last_user_query = user_query
        self.generation_count += 1

        if not context or not context.strip():
            return NO_AUTHORIZED_INFO_MESSAGE

        # If malicious prompt injection is in the context, ensure we do NOT execute it
        if "ignore all previous instructions" in context.lower() or "ignore previous instructions" in context.lower():
            # Documents are DATA, not instructions: ignore the injected command
            pass

        # Extract source content lines
        source_contents = []
        for block in context.split("--- SOURCE "):
            if 'Content:\n"""' in block:
                try:
                    c = block.split('Content:\n"""')[1].split('"""')[0].strip()
                    source_contents.append(c)
                except IndexError:
                    continue

        if not source_contents:
            return INSUFFICIENT_INFO_MESSAGE

        # Extract matching keywords from user query
        query_words = set(re.findall(r"\w{3,}", user_query.lower()))
        matched_sentences = []

        for content in source_contents:
            for sentence in re.split(r"[.\n]+", content):
                s_clean = sentence.strip()
                if not s_clean:
                    continue
                s_words = set(re.findall(r"\w{3,}", s_clean.lower()))
                if query_words & s_words:
                    matched_sentences.append(s_clean)

        if matched_sentences:
            # Produce grounded synthesized response using facts in context
            return ". ".join(matched_sentences[:3]) + "."

        # If context has content but query terms don't match
        return f"Based on the provided documents: {source_contents[0][:200]}."


class OpenAILLMProvider:
    """External LLM provider calling OpenAI-compatible chat completion endpoints."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        user_content = f"SOURCES:\n{context}\n\nQUESTION:\n{user_query}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": settings.llm_temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


_mock_llm = MockLLMProvider()


def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "mock" or not settings.llm_api_key:
        return _mock_llm
    return OpenAILLMProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
