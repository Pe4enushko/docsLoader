from __future__ import annotations

import re

from app.schemas.visit import CanonicalVisit


class DiagnosisContextExtractor:
    ICD_PATTERN = re.compile(r"\b([A-ZА-Я]\d{2}(?:\.\d+)?)\b", re.IGNORECASE)

    def extract_icd10_codes(self, visit: CanonicalVisit) -> list[str]:
        assessment_blob = " ".join(str(v) for v in visit.assessment.values())
        codes = self.ICD_PATTERN.findall(assessment_blob.upper())
        return sorted(set(codes))

    def extract_primary_diagnosis_text(self, visit: CanonicalVisit) -> str:
        if not visit.assessment:
            return ""

        for key in ("diagnosis", "diag", "диагноз", "основной_диагноз"):
            for existing_key, value in visit.assessment.items():
                if key in existing_key.lower():
                    return str(value)

        return " ".join(str(v) for v in visit.assessment.values())
