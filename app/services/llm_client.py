from __future__ import annotations

"""LLM client abstractions and OpenAI-compatible implementation.

This module exposes a narrow interface (`LLMClient`) used by pipelines and
extractors. Implementation details (OpenAI endpoint, model, timeout) stay here.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from app.config import get_settings
from app.schemas.llm import LLMResponse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class LLMClient(ABC):
    """Generic chat-completion contract used by the application."""

    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str, model: str | None = None) -> LLMResponse:
        """Generate a text response for a `(system, user)` prompt pair."""
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible chat client.

    Works with OpenAI API and OpenAI-compatible gateways by supporting custom
    `base_url` and model names from environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialize transport client and defaults.

        `api_key` can be empty when using local gateways that ignore auth.
        """
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required. Install: pip install openai")

        settings = get_settings()
        resolved_key = api_key if api_key is not None else settings.openai_api_key
        resolved_base_url = base_url if base_url is not None else settings.openai_llm_base_url

        self.default_model = default_model or settings.llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.client = OpenAI(api_key=resolved_key or "EMPTY", base_url=resolved_base_url, timeout=self.timeout_seconds)

    def generate(self, *, system_prompt: str, user_prompt: str, model: str | None = None) -> LLMResponse:
        """Execute chat completion and normalize output to internal schema."""
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

        return LLMResponse(
            text=text,
            raw=raw_payload,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )
