from __future__ import annotations

from app.domain.enums import LLMCheckStage
from app.prompts.base import StagePromptBuilder
from app.prompts.diagnosis_consistency import DiagnosisConsistencyPromptBuilder
from app.prompts.documentation_quality import DocumentationQualityPromptBuilder
from app.prompts.followup import FollowupPromptBuilder
from app.prompts.formal_structure import FormalStructurePromptBuilder
from app.prompts.management_consistency import ManagementConsistencyPromptBuilder


class PromptBuilderRegistry:
    def __init__(self) -> None:
        self._builders: dict[LLMCheckStage, StagePromptBuilder] = {
            LLMCheckStage.FORMAL_STRUCTURE_CHECK: FormalStructurePromptBuilder(),
            LLMCheckStage.DIAGNOSIS_CONSISTENCY_CHECK: DiagnosisConsistencyPromptBuilder(),
            LLMCheckStage.MANAGEMENT_CONSISTENCY_CHECK: ManagementConsistencyPromptBuilder(),
            LLMCheckStage.FOLLOWUP_CHECK: FollowupPromptBuilder(),
            LLMCheckStage.DOCUMENTATION_QUALITY_CHECK: DocumentationQualityPromptBuilder(),
        }

    def get(self, stage: LLMCheckStage) -> StagePromptBuilder:
        return self._builders[stage]

    def stages(self) -> list[LLMCheckStage]:
        return list(self._builders.keys())
