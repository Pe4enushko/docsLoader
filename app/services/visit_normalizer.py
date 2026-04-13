from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.schemas.visit import CanonicalVisit


KEYS_MAP: dict[str, tuple[str, ...]] = {
    "meta": ("meta", "metadata", "visit_meta"),
    "patient": ("patient", "пациент"),
    "subjective": ("subjective", "complaints", "жалобы", "анамнез"),
    "objective": ("objective", "exam", "осмотр", "status"),
    "assessment": ("assessment", "diagnosis", "diagnoses", "оценка", "диагноз"),
    "plan": ("plan", "treatment", "recommendations", "план", "рекомендации"),
    "admin": ("admin", "administrative", "service", "услуга"),
}


class VisitNormalizer:
    def normalize(self, payload: Mapping[str, Any]) -> CanonicalVisit:
        normalized = {}
        for canonical_key, candidates in KEYS_MAP.items():
            normalized[canonical_key] = self._extract_first(payload, candidates)
            if not isinstance(normalized[canonical_key], dict):
                normalized[canonical_key] = {}

        if not normalized["assessment"]:
            normalized["assessment"] = self._fallback_assessment(payload)

        return CanonicalVisit(**normalized)

    def _extract_first(self, payload: Mapping[str, Any], candidates: tuple[str, ...]) -> dict[str, Any]:
        lowered = {str(key).lower(): key for key in payload.keys()}
        for candidate in candidates:
            key = lowered.get(candidate.lower())
            if key is not None:
                value = payload.get(key)
                if isinstance(value, Mapping):
                    return dict(value)
        return {}

    def _fallback_assessment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        assessment: dict[str, Any] = {}
        for key in payload:
            key_lower = str(key).lower()
            if "mkb" in key_lower or "icd" in key_lower or "diag" in key_lower or "диаг" in key_lower:
                assessment[str(key)] = payload[key]
        return assessment
