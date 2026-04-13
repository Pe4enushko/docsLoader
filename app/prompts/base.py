from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.enums import LLMCheckStage
from app.schemas.llm import StagePrompt
from app.schemas.retrieval import RetrievalContext
from app.schemas.visit import CanonicalVisit


class StagePromptBuilder(ABC):
    stage: LLMCheckStage
    version: str = "v1"

    @abstractmethod
    def build(
        self,
        visit: CanonicalVisit,
        retrieval_context: RetrievalContext,
        prompt_conditions: list[str],
    ) -> StagePrompt:
        raise NotImplementedError
