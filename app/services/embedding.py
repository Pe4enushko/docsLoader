from __future__ import annotations

"""Embedding provider abstractions and OpenAI-compatible implementation."""

from abc import ABC, abstractmethod

from app.config import get_settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding backends."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert list of input texts into embedding vectors."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings provider.

    Uses model and base URL from environment, so the same code works for
    OpenAI cloud and self-hosted OpenAI-compatible services.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required. Install: pip install openai")

        settings = get_settings()
        resolved_key = api_key if api_key is not None else settings.openai_api_key
        resolved_base_url = base_url if base_url is not None else settings.openai_embedding_base_url

        self.model = model or settings.embedding_model
        self.dimensions = dimensions if dimensions is not None else settings.embedding_dimension
        self.client = OpenAI(api_key=resolved_key or "EMPTY", base_url=resolved_base_url, timeout=timeout_seconds or settings.llm_timeout_seconds)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings in batch mode with graceful dimensions fallback."""
        if not texts:
            return []

        kwargs = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            kwargs["dimensions"] = self.dimensions

        try:
            response = self.client.embeddings.create(**kwargs)
        except Exception:
            # Some compatible providers ignore/forbid `dimensions`.
            kwargs.pop("dimensions", None)
            response = self.client.embeddings.create(**kwargs)

        return [item.embedding for item in response.data]
