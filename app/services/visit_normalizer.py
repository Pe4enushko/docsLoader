from __future__ import annotations

import re
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

PLAN_PARAM_MARKERS = (
    "рекомендац",
    "назнач",
    "план",
    "тактик",
    "консультац",
    "лечение",
    "направлен",
    "контроль",
    "follow",
)

SUBJECTIVE_PARAM_MARKERS = (
    "жалоб",
    "анамнез",
    "динамик",
    "со слов",
    "самочув",
    "беспок",
)


def _normalize_key(value: str) -> str:
    """Normalize key for case-insensitive and locale-tolerant matching."""
    return str(value).strip().lower().replace("ё", "е")


class VisitNormalizer:
    """Converts heterogeneous raw visit payloads into canonical schema sections."""

    def normalize(self, payload: Mapping[str, Any]) -> CanonicalVisit:
        """Normalize one visit payload.

        Supports two main formats:
        1) already partially-canonical JSON with blocks like patient/assessment/plan;
        2) one element from `appointments[]` 1C payload with Russian keys.
        """
        appointment_item = self._extract_appointment_item(payload)
        if appointment_item is not None:
            return self._normalize_appointment_item(appointment_item)

        normalized = {}
        for canonical_key, candidates in KEYS_MAP.items():
            normalized[canonical_key] = self._extract_first(payload, candidates)
            if not isinstance(normalized[canonical_key], dict):
                normalized[canonical_key] = {}

        if not normalized["assessment"]:
            normalized["assessment"] = self._fallback_assessment(payload)

        return CanonicalVisit(**normalized)

    def _extract_first(self, payload: Mapping[str, Any], candidates: tuple[str, ...]) -> dict[str, Any]:
        """Extract first matching dict block by alias list."""
        lowered = {str(key).lower(): key for key in payload.keys()}
        for candidate in candidates:
            key = lowered.get(candidate.lower())
            if key is not None:
                value = payload.get(key)
                if isinstance(value, Mapping):
                    return dict(value)
        return {}

    def _fallback_assessment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Fallback diagnosis extraction when canonical assessment block is absent."""
        assessment: dict[str, Any] = {}
        for key in payload:
            key_lower = str(key).lower()
            if "mkb" in key_lower or "icd" in key_lower or "diag" in key_lower or "диаг" in key_lower:
                assessment[str(key)] = payload[key]
        return assessment

    def _extract_appointment_item(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Detect and return one 1C appointment item payload."""
        payload_dict = dict(payload)
        if self._looks_like_appointment_item(payload_dict):
            return payload_dict

        appointments = self._extract_by_alias(payload_dict, ("appointments", "appointment"))
        if isinstance(appointments, list):
            valid_items = [item for item in appointments if isinstance(item, Mapping)]
            if len(valid_items) == 1:
                item = dict(valid_items[0])
                if self._looks_like_appointment_item(item):
                    return item
        return None

    def _looks_like_appointment_item(self, payload: Mapping[str, Any]) -> bool:
        """Heuristic detection of one appointment element from 1C response."""
        markers = {
            "прием",
            "приём",
            "врач",
            "пациент",
            "услуги",
            "данныеосмотра",
            "диагнозы",
        }
        found = 0
        for key in payload.keys():
            if _normalize_key(str(key)) in markers:
                found += 1
        return found >= 2

    def _normalize_appointment_item(self, payload: Mapping[str, Any]) -> CanonicalVisit:
        """Normalize one element of `appointments[]` into canonical visit sections."""
        visit_block = self._extract_dict_by_alias(payload, ("Прием", "Приём", "visit"))
        doctor_block = self._extract_dict_by_alias(payload, ("Врач", "doctor", "physician"))
        patient_block = self._extract_dict_by_alias(payload, ("Пациент", "patient"))

        services = self._extract_list_of_dicts(payload, ("Услуги", "services", "service"))
        exam_rows = self._extract_list_of_dicts(payload, ("ДанныеОсмотра", "осмотр", "exam_data"))
        diagnoses = self._extract_list_of_dicts(payload, ("Диагнозы", "diagnoses", "diagnosis"))

        meta = self._build_meta_section(visit_block=visit_block, doctor_block=doctor_block, services=services)
        patient = self._build_patient_section(patient_block)
        subjective, objective, plan = self._split_exam_rows(exam_rows)
        assessment = self._build_assessment_section(diagnoses=diagnoses, exam_rows=exam_rows)
        admin = self._build_admin_section(doctor_block=doctor_block, services=services, visit_block=visit_block)

        if not assessment:
            assessment = self._fallback_assessment(payload)

        return CanonicalVisit(
            meta=meta,
            patient=patient,
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            admin=admin,
        )

    def _build_meta_section(
        self,
        *,
        visit_block: dict[str, Any],
        doctor_block: dict[str, Any],
        services: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build canonical `meta` payload with visit identifiers and source hints."""
        guid = self._extract_by_alias(visit_block, ("GUID", "guid"))
        visit_date = self._extract_by_alias(visit_block, ("DATE", "date"))
        specialty = self._extract_by_alias(doctor_block, ("SPECIALIZATION", "specialization", "specialty"))

        service_names = []
        for item in services:
            name = self._extract_by_alias(item, ("Наименование", "НаименованиеЕГИСЗ", "name"))
            if name:
                service_names.append(str(name))

        return {
            "source_system": "1c_appointments",
            "visit_guid": str(guid).strip() if guid is not None else "",
            "visit_date": str(visit_date).strip() if visit_date is not None else "",
            "specialty": str(specialty).strip() if specialty is not None else "",
            "service_count": len(services),
            "service_names": service_names,
        }

    def _build_patient_section(self, patient_block: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize patient demographics into stable canonical keys."""
        result: dict[str, Any] = {}
        consumed_source_keys: set[str] = set()
        aliases = {
            "age": ("AGE", "age", "Возраст", "возраст"),
            "gender": ("GENDER", "gender", "Пол", "пол"),
            "birth_date": ("BIRTHDATE", "birth_date", "датарождения", "дата рождения"),
        }
        for canonical_key, candidates in aliases.items():
            source_key = self._find_key_by_alias(patient_block, candidates)
            value = patient_block.get(source_key) if source_key is not None else None
            if value is not None and str(value).strip():
                result[canonical_key] = value
                consumed_source_keys.add(_normalize_key(source_key))

        for key, value in patient_block.items():
            if _normalize_key(key) in consumed_source_keys:
                continue
            if _normalize_key(key) not in {"age", "gender", "birth_date"}:
                result[str(key)] = value
        return result

    def _build_assessment_section(
        self,
        *,
        diagnoses: list[dict[str, Any]],
        exam_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Convert diagnosis list into canonical assessment structure."""
        result: dict[str, Any] = {}
        codes: list[str] = []
        diagnosis_texts: list[str] = []

        for index, diagnosis in enumerate(diagnoses, start=1):
            code = self._extract_by_alias(diagnosis, ("КодМКБ", "кодмкб", "icd10", "icd", "code"))
            title = self._extract_by_alias(diagnosis, ("НаименованиеМКБ", "наименование", "diagnosis", "name"))
            details = self._extract_by_alias(diagnosis, ("Детализация", "details", "comment"))
            first_detected = self._extract_by_alias(diagnosis, ("ВыявленВпервые", "first_detected"))
            first_detected_value = self._coerce_optional_bool(first_detected)

            entry = {
                "code": str(code).strip() if code is not None else "",
                "title": str(title).strip() if title is not None else "",
                "details": str(details).strip() if details is not None else "",
                "first_detected": first_detected_value,
            }
            result[f"diagnosis_{index}"] = entry

            if entry["code"]:
                codes.append(entry["code"])
            text_parts = [entry["code"], entry["title"], entry["details"]]
            merged_text = " ".join(part for part in text_parts if part).strip()
            if merged_text:
                diagnosis_texts.append(merged_text)

        for row in exam_rows:
            param = str(self._extract_by_alias(row, ("Параметр", "parameter", "name")) or "").strip()
            value = str(self._extract_by_alias(row, ("Значение", "value", "text")) or "").strip()
            if "диагноз" in _normalize_key(param) and value:
                diagnosis_texts.append(value)

        if codes:
            result["icd10_codes"] = sorted(set(codes))
        if diagnosis_texts:
            result["diagnosis_text"] = " | ".join(diagnosis_texts)
        if result.get("diagnosis_1"):
            result["primary_diagnosis"] = result["diagnosis_1"]
        return result

    def _build_admin_section(
        self,
        *,
        doctor_block: Mapping[str, Any],
        services: list[dict[str, Any]],
        visit_block: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Collect service/provider metadata used by downstream classification."""
        specialty = self._extract_by_alias(doctor_block, ("SPECIALIZATION", "specialization", "specialty"))
        service_names: list[str] = []
        service_codes: list[str] = []
        egisz_codes: list[str] = []
        service_uids: list[str] = []

        for service in services:
            name = self._extract_by_alias(service, ("Наименование", "НаименованиеЕГИСЗ", "name"))
            if name:
                service_names.append(str(name))

            code = self._extract_by_alias(service, ("Код", "code", "Артикул", "article"))
            if code:
                service_codes.append(str(code))

            egisz = self._extract_by_alias(service, ("КодЕГИСЗ", "егисз", "egisz_code"))
            if egisz:
                egisz_codes.append(str(egisz))

            uid = self._extract_by_alias(service, ("УИДЕГИСЗ", "uidegisz", "egisz_uid"))
            if uid is not None:
                service_uids.append(str(uid))

        return {
            "specialty": str(specialty).strip() if specialty is not None else "",
            "service_names": service_names,
            "service_text": " | ".join(service_names),
            "service_codes": service_codes,
            "egisz_codes": egisz_codes,
            "egisz_uids": service_uids,
            "visit_guid": str(self._extract_by_alias(visit_block, ("GUID", "guid")) or "").strip(),
        }

    def _split_exam_rows(self, exam_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Split exam parameters into subjective/objective/plan canonical blocks."""
        subjective: dict[str, Any] = {}
        objective: dict[str, Any] = {}
        plan: dict[str, Any] = {}

        for index, row in enumerate(exam_rows, start=1):
            param = str(self._extract_by_alias(row, ("Параметр", "parameter", "name")) or "").strip()
            value = str(self._extract_by_alias(row, ("Значение", "value", "text")) or "").strip()
            if not (param or value):
                continue

            target = self._classify_exam_param(param)
            key = self._make_param_key(param, index)
            destination = subjective if target == "subjective" else plan if target == "plan" else objective
            destination[self._ensure_unique_key(destination, key)] = value
        return subjective, objective, plan

    def _classify_exam_param(self, param: str) -> str:
        """Classify exam row target section by parameter name heuristics."""
        normalized = _normalize_key(param)
        if any(marker in normalized for marker in PLAN_PARAM_MARKERS):
            return "plan"
        if any(marker in normalized for marker in SUBJECTIVE_PARAM_MARKERS):
            return "subjective"
        return "objective"

    def _make_param_key(self, param: str, index: int) -> str:
        """Generate stable machine-readable key from free-form exam parameter name."""
        normalized = _normalize_key(param)
        normalized = normalized.replace("-", " ").replace("/", " ")
        normalized = re.sub(r"\s+", "_", normalized)
        normalized = re.sub(r"[^0-9a-zа-я_]+", "", normalized)
        normalized = normalized.strip("_")
        if not normalized:
            return f"exam_field_{index}"
        return normalized

    def _ensure_unique_key(self, target: dict[str, Any], key: str) -> str:
        """Prevent collisions for repeated parameter names."""
        if key not in target:
            return key
        suffix = 2
        while f"{key}_{suffix}" in target:
            suffix += 1
        return f"{key}_{suffix}"

    def _extract_dict_by_alias(self, payload: Mapping[str, Any], aliases: tuple[str, ...]) -> dict[str, Any]:
        """Extract dict value by key aliases (case-insensitive)."""
        value = self._extract_by_alias(payload, aliases)
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _extract_list_of_dicts(self, payload: Mapping[str, Any], aliases: tuple[str, ...]) -> list[dict[str, Any]]:
        """Extract list[dict] value by aliases, filtering malformed entries."""
        value = self._extract_by_alias(payload, aliases)
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, Mapping)]

    def _extract_by_alias(self, payload: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
        """Extract any value by key aliases with locale-normalized lookup."""
        original_key = self._find_key_by_alias(payload, aliases)
        if original_key is not None:
            return payload.get(original_key)
        return None

    def _find_key_by_alias(self, payload: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
        """Find original key name by aliases using normalized matching."""
        keys_index = {_normalize_key(str(key)): key for key in payload.keys()}
        for alias in aliases:
            original_key = keys_index.get(_normalize_key(alias))
            if original_key is not None:
                return str(original_key)
        return None

    def _coerce_optional_bool(self, value: Any) -> bool | None:
        """Convert loose boolean-like values to bool while preserving unknowns."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        normalized = _normalize_key(str(value))
        if normalized in {"true", "1", "yes", "да"}:
            return True
        if normalized in {"false", "0", "no", "нет"}:
            return False
        return None
