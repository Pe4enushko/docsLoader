from __future__ import annotations

import re
from collections import Counter

from app.schemas.visit import CanonicalVisit, HeuristicFlag


PLACEHOLDER_VALUES = {"-", "n/a", "не заполнено", "нет данных", "test", "sample"}


class VisitHeuristicFlagger:
    def generate_flags(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        flags: list[HeuristicFlag] = []

        required_blocks = ["patient", "assessment", "plan"]
        for block in required_blocks:
            value = getattr(visit, block)
            if not value:
                flags.append(
                    HeuristicFlag(
                        code="missing_required_field",
                        severity="error",
                        message=f"Missing required block: {block}",
                        field_path=block,
                    )
                )

        flags.extend(self._detect_placeholders(visit))
        flags.extend(self._detect_duplicate_parameters(visit))
        flags.extend(self._detect_plan_result_mixup(visit))
        flags.extend(self._detect_dx_too_general(visit))
        flags.extend(self._detect_recommendation_without_basis(visit))

        return flags

    def _detect_placeholders(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        result: list[HeuristicFlag] = []
        for section_name in ("subjective", "objective", "assessment", "plan"):
            section = getattr(visit, section_name)
            for key, value in section.items():
                if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
                    result.append(
                        HeuristicFlag(
                            code="placeholder_value",
                            message=f"Placeholder value in {section_name}.{key}",
                            field_path=f"{section_name}.{key}",
                        )
                    )
        return result

    def _detect_duplicate_parameters(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        flags: list[HeuristicFlag] = []
        all_keys: list[str] = []
        for section_name in ("subjective", "objective", "assessment", "plan"):
            all_keys.extend(f"{section_name}.{key.lower()}" for key in getattr(visit, section_name).keys())

        base_key_count = Counter(key.split(".", 1)[1] for key in all_keys)
        for key, count in base_key_count.items():
            if count > 1:
                flags.append(
                    HeuristicFlag(
                        code="duplicate_parameter",
                        message=f"Duplicate parameter across sections: {key}",
                    )
                )
        return flags

    def _detect_plan_result_mixup(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        flags: list[HeuristicFlag] = []
        plan_blob = " ".join(str(v) for v in visit.plan.values()).lower()
        objective_blob = " ".join(str(v) for v in visit.objective.values()).lower()

        if re.search(r"(гемоглобин|лейкоцит|анализ|ммоль|мг/л)", plan_blob):
            flags.append(
                HeuristicFlag(
                    code="result_in_plan_field",
                    message="Lab/result-like values appear in plan section",
                    field_path="plan",
                )
            )
        if re.search(r"(назнач|рекоменд|повторн|контроль)", objective_blob):
            flags.append(
                HeuristicFlag(
                    code="plan_in_result_field",
                    message="Plan-like action appears in objective section",
                    field_path="objective",
                )
            )
        return flags

    def _detect_dx_too_general(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        assessment_blob = " ".join(str(v) for v in visit.assessment.values()).lower()
        if any(marker in assessment_blob for marker in ("орви", "состояние", "жалобы")) and not re.search(
            r"\b[A-ZА-Я]\d{2}(?:\.\d+)?\b", assessment_blob.upper()
        ):
            return [
                HeuristicFlag(
                    code="dx_too_general",
                    message="Diagnosis may be too general and lacks ICD-like specificity",
                    field_path="assessment",
                )
            ]
        return []

    def _detect_recommendation_without_basis(self, visit: CanonicalVisit) -> list[HeuristicFlag]:
        plan_blob = " ".join(str(v) for v in visit.plan.values()).strip()
        subjective_blob = " ".join(str(v) for v in visit.subjective.values()).strip()
        objective_blob = " ".join(str(v) for v in visit.objective.values()).strip()

        if plan_blob and not (subjective_blob or objective_blob):
            return [
                HeuristicFlag(
                    code="recommendation_without_basis",
                    message="Plan exists without subjective/objective basis",
                    field_path="plan",
                )
            ]
        return []
