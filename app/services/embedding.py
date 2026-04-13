from __future__ import annotations

"""Embedding adapter layer with pluggable backends.

Pattern used in this module:
- `RawEmbeddingBackend` defines provider-specific embedding contract.
- Backend classes implement transport specifics (`OpenAIEmbeddingsBackend`, `OllamaEmbeddingsBackend`).
- `EmbeddingsAdapter` exposes normalized `EmbeddingProvider` API used by pipelines.
"""

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
    """Target interface consumed by ingestion/retrieval pipelines."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert list of input texts into embedding vectors."""
        raise NotImplementedError


class RawEmbeddingBackend(ABC):
    """Provider-specific low-level embedding backend contract."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for input texts."""
        raise NotImplementedError


class OpenAIEmbeddingsBackend(RawEmbeddingBackend):
    """Raw backend for OpenAI-compatible embeddings API."""

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

    def embed(self, texts: list[str]) -> list[list[float]]:
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


class OllamaEmbeddingsBackend(RawEmbeddingBackend):
    """Raw backend for Ollama embeddings API."""

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

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_embeddings = self._try_batch_embed(texts)
        if batch_embeddings is not None:
            return batch_embeddings

        # Fallback for old Ollama versions without `/api/embed`.
        return [self._embed_single(text) for text in texts]

    def _try_batch_embed(self, texts: list[str]) -> list[list[float]] | None:
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

        single = data.get("embedding")
        if isinstance(single, list):
            return [single]

        raise RuntimeError(f"Unexpected Ollama embeddings payload: {self._preview(data)}")

    def _embed_single(self, text: str) -> list[float]:
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
        text = str(payload)
        return text[:350] + ("..." if len(text) > 350 else "")


class EmbeddingsAdapter(EmbeddingProvider):
    """Adapter that maps raw backend to unified `EmbeddingProvider` API."""

    def __init__(self, *, backend: RawEmbeddingBackend) -> None:
        self.backend = backend

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed(texts)


class OpenAIEmbeddingProvider(EmbeddingsAdapter):
    """Backward-compatible OpenAI embedding provider adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        backend = OpenAIEmbeddingsBackend(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(backend=backend)


class OllamaEmbeddings(EmbeddingsAdapter):
    """Backward-compatible Ollama embedding provider adapter."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        backend = OllamaEmbeddingsBackend(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(backend=backend)


def create_embedding_backend(provider_name: str | None = None) -> RawEmbeddingBackend:
    """Factory for raw embedding backend selection from env/config."""
    settings = get_settings()
    name = (provider_name or settings.embedding_provider).strip().lower()

    if name == "openai":
        return OpenAIEmbeddingsBackend()
    if name == "ollama":
        return OllamaEmbeddingsBackend()

    raise ValueError(f"Unsupported embedding provider: {name}. Expected 'openai' or 'ollama'.")


def create_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    """Factory for adapter embedding provider selection from env/config."""
    settings = get_settings()
    name = (provider_name or settings.embedding_provider).strip().lower()

    if name == "openai":
        return OpenAIEmbeddingProvider()
    if name == "ollama":
        return OllamaEmbeddings()

    raise ValueError(f"Unsupported embedding provider: {name}. Expected 'openai' or 'ollama'.")
