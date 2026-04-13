from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.enums import LLMCheckStage


class StagePrompt(BaseModel):
    stage: LLMCheckStage
    prompt_version: str = "v1"
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageCheckResult(BaseModel):
    stage: LLMCheckStage
    status: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    summary: str
    raw_response: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0


class LLMResponse(BaseModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
