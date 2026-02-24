from __future__ import annotations

import json
from typing import Any


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("ё", "е")


def _pick_key(data: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    wanted = {_normalize_key(name) for name in candidates}
    for key in data.keys():
        if _normalize_key(str(key)) in wanted:
            return str(key)
    return None


def _to_json_obj(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        return json.loads(text)
    raise ValueError(f"Unsupported appointments payload type: {type(raw).__name__}")


def extract_visit_dict(appointment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(appointment, dict):
        return {}

    visit_key = _pick_key(appointment, ("Прием", "Приём", "visit"))
    if visit_key and isinstance(appointment.get(visit_key), dict):
        return appointment[visit_key]

    if "GUID" in appointment or "DATE" in appointment:
        return appointment

    return {}


def extract_visit_guid(appointment: dict[str, Any]) -> str:
    visit = extract_visit_dict(appointment)
    guid_key = _pick_key(visit, ("GUID", "guid"))
    if not guid_key:
        return ""
    return str(visit.get(guid_key, "")).strip()


def extract_visit_date_raw(appointment: dict[str, Any]) -> str:
    visit = extract_visit_dict(appointment)
    date_key = _pick_key(visit, ("DATE", "date"))
    if not date_key:
        return ""
    return str(visit.get(date_key, "")).strip()


def _extract_appointments_container(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Appointments payload must be JSON object or array")

    appointments_key = _pick_key(payload, ("appointments", "appointment"))
    if appointments_key:
        value = payload.get(appointments_key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        raise ValueError(f"Field '{appointments_key}' must be array or object")

    data_key = _pick_key(payload, ("data", "result", "response"))
    if data_key:
        return _extract_appointments_container(payload.get(data_key))

    return [payload]


def parse_appointments_payload(payload: Any) -> list[dict[str, Any]]:
    obj = _to_json_obj(payload)
    raw_items = _extract_appointments_container(obj)

    appointments: list[dict[str, Any]] = []
    for item in raw_items:
        item_obj = _to_json_obj(item)
        if not isinstance(item_obj, dict):
            continue

        if extract_visit_guid(item_obj):
            appointments.append(item_obj)
            continue

        nested_key = _pick_key(item_obj, ("appointment", "data", "result", "record"))
        nested = item_obj.get(nested_key) if nested_key else None
        if isinstance(nested, dict) and extract_visit_guid(nested):
            appointments.append(nested)
            continue

        appointments.append(item_obj)
    return appointments
