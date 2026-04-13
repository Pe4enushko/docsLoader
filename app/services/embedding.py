from __future__ import annotations

"""Embedding provider abstractions and OpenAI-compatible implementation."""

from abc import ABC, abstractmethod
from typing import Any

import requests

from app.config import get_settings
from app.utils.logging import get_logger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

log = get_logger(__name__)


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
        self.client = OpenAI(
            api_key=resolved_key or "EMPTY",
            base_url=resolved_base_url,
            timeout=timeout_seconds or settings.embedding_timeout_seconds,
        )

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


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embeddings backend.

    Supports both modern `/api/embed` batch endpoint and legacy
    `/api/embeddings` single-input endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_embed_base_url).rstrip("/")
        self.model = model or settings.ollama_embed_model
        self.timeout_seconds = timeout_seconds or settings.embedding_timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed list of texts using Ollama HTTP API."""
        if not texts:
            return []

        batch_embeddings = self._try_batch_embed(texts)
        if batch_embeddings is not None:
            return batch_embeddings

        # Fallback for old Ollama versions without `/api/embed`.
        return [self._embed_single(text) for text in texts]

    def _try_batch_embed(self, texts: list[str]) -> list[list[float]] | None:
        """Try `/api/embed` endpoint; return `None` when unsupported."""
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}

        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if response.status_code == 404:
            log.info("Ollama batch endpoint not available, switching to legacy mode | url=%s", url)
            return None

        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings

        # Some servers may return single embedding shape.
        single = data.get("embedding")
        if isinstance(single, list):
            return [single]

        raise RuntimeError(f"Unexpected Ollama embeddings payload: {self._preview(data)}")

    def _embed_single(self, text: str) -> list[float]:
        """Embed one text using legacy `/api/embeddings` endpoint."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
        raise RuntimeError(f"Unexpected Ollama embedding payload: {self._preview(data)}")

    def _preview(self, payload: Any) -> str:
        """Create short payload preview for logs/errors."""
        text = str(payload)
        return text[:350] + ("..." if len(text) > 350 else "")


def create_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    """Factory for embedding provider selection from config/env."""
    settings = get_settings()
    name = (provider_name or settings.embedding_provider).strip().lower()

    if name == "openai":
        return OpenAIEmbeddingProvider()
    if name == "ollama":
        return OllamaEmbeddings()

    raise ValueError(f"Unsupported embedding provider: {name}. Expected 'openai' or 'ollama'.")
