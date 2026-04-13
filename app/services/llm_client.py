from __future__ import annotations

"""LLM adapter layer with pluggable backends.

Pattern used in this module:
- `RawLLMBackend` defines provider-specific invocation contract.
- Backend classes (`OpenAIChatBackend`, `OllamaChatBackend`) implement transport specifics.
- `LLMAdapter` normalizes backend output into `LLMResponse` used by pipelines.

This makes backend replacement possible without changing audit or ingestion pipelines.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from app.config import get_settings
from app.schemas.llm import LLMResponse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class LLMClient(ABC):
    """Target interface consumed by pipelines and services."""

    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str, model: str | None = None) -> LLMResponse:
        """Generate text for `(system, user)` prompt pair."""
        raise NotImplementedError


class RawLLMBackend(ABC):
    """Provider-specific low-level LLM backend contract."""

    @abstractmethod
    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, int], int]:
        """Return `(text, raw_payload, token_usage, latency_ms)` from provider."""
        raise NotImplementedError


class OpenAIChatBackend(RawLLMBackend):
    """Raw backend for OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required. Install: pip install openai")

        settings = get_settings()
        resolved_key = api_key if api_key is not None else settings.openai_api_key
        resolved_base_url = base_url if base_url is not None else settings.openai_llm_base_url

        self.default_model = default_model or settings.llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.client = OpenAI(api_key=resolved_key or "EMPTY", base_url=resolved_base_url, timeout=self.timeout_seconds)

    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, int], int]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = response.choices[0].message.content if response.choices else ""
        text = message or ""

        usage = response.usage
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

        raw_payload: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_payload = response.model_dump()  # type: ignore[assignment]
        else:  # pragma: no cover
            raw_payload = json.loads(json.dumps(response, default=str))

        return text, raw_payload, token_usage, latency_ms


class OllamaChatBackend(RawLLMBackend):
    """Raw backend for Ollama `/api/chat` endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
        num_ctx: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_llm_base_url).rstrip("/")
        self.default_model = default_model or settings.ollama_llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.num_ctx = settings.ollama_llm_num_ctx if num_ctx is None else num_ctx

    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, int], int]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.num_ctx > 0:
            payload["options"] = {"num_ctx": self.num_ctx}

        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        raw_payload = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = str((raw_payload.get("message") or {}).get("content", "") or "")
        prompt_tokens = int(raw_payload.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(raw_payload.get("eval_count", 0) or 0)
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text, raw_payload, token_usage, latency_ms


class LLMAdapter(LLMClient):
    """Adapter that maps raw backend response to normalized `LLMResponse`."""

    def __init__(self, *, backend: RawLLMBackend, default_model: str | None = None) -> None:
        self.backend = backend
        self.default_model = default_model

    def generate(self, *, system_prompt: str, user_prompt: str, model: str | None = None) -> LLMResponse:
        text, raw, token_usage, latency_ms = self.backend.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or self.default_model,
        )
        return LLMResponse(
            text=text,
            raw=raw,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )


class OpenAILLMClient(LLMAdapter):
    """Backward-compatible OpenAI adapter client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        backend = OpenAIChatBackend(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(backend=backend, default_model=default_model)


class OllamaLLMClient(LLMAdapter):
    """Backward-compatible Ollama adapter client."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
        num_ctx: int | None = None,
    ) -> None:
        backend = OllamaChatBackend(
            base_url=base_url,
            default_model=default_model,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
        )
        super().__init__(backend=backend, default_model=default_model)


def create_llm_backend(provider_name: str | None = None, *, default_model: str | None = None) -> RawLLMBackend:
    """Factory for raw backend selection from env/config."""
    settings = get_settings()
    name = (provider_name or settings.llm_provider).strip().lower()

    if name == "openai":
        return OpenAIChatBackend(default_model=default_model)
    if name == "ollama":
        return OllamaChatBackend(default_model=default_model)

    raise ValueError(f"Unsupported LLM provider: {name}. Expected 'openai' or 'ollama'.")


def create_llm_client(provider_name: str | None = None, *, default_model: str | None = None) -> LLMClient:
    """Factory for adapter client selection from env/config."""
    settings = get_settings()
    name = (provider_name or settings.llm_provider).strip().lower()

    if name == "openai":
        return OpenAILLMClient(default_model=default_model)
    if name == "ollama":
        return OllamaLLMClient(default_model=default_model)

    raise ValueError(f"Unsupported LLM provider: {name}. Expected 'openai' or 'ollama'.")
