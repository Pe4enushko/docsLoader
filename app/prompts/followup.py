from __future__ import annotations

from app.domain.enums import LLMCheckStage
from app.prompts.base import StagePromptBuilder
from app.prompts.common import render_conditions, render_retrieval_context, render_visit_compact
from app.schemas.llm import StagePrompt
from app.schemas.retrieval import RetrievalContext
from app.schemas.visit import CanonicalVisit


SYSTEM_PROMPT = """You audit follow-up adequacy in medical visit documentation.
Return JSON with keys: status, summary, findings[] and focus on monitoring intervals, red flags, and next-step clarity."""


class FollowupPromptBuilder(StagePromptBuilder):
    stage = LLMCheckStage.FOLLOWUP_CHECK

    def build(
        self,
        visit: CanonicalVisit,
        retrieval_context: RetrievalContext,
        prompt_conditions: list[str],
    ) -> StagePrompt:
        user_prompt = "\n\n".join(
            [
                "Check follow-up adequacy.",
                "### Conditions",
                render_conditions(prompt_conditions),
                "### Visit",
                render_visit_compact(visit),
                render_retrieval_context(retrieval_context),
            ]
        )
        return StagePrompt(
            stage=self.stage,
            prompt_version=self.version,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
