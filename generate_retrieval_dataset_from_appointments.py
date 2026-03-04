from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from engine.config import Settings
from engine.llm import AppointmentJudge, normalize_mkb_code
from engine.logging_utils import setup_logging

INPUT_JSON = os.getenv("APPOINTMENTS_JSON", "parseddata.json")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.csv")
OUT_TXT = os.getenv("OUT_RETRIEVAL_TXT", "retrieval_test_queries.generated.txt")
OUT_JSON = os.getenv("OUT_RETRIEVAL_JSON", "retrieval_dataset.generated.json")
LOG_FILE = os.getenv("GEN_DATASET_LOG", "logs/generate_retrieval_dataset.log")
MAX_APPOINTMENTS = int(os.getenv("MAX_APPOINTMENTS", "300"))


class _NoopRetrieval:
    """AppointmentJudge requires retrieval in constructor, but query generation doesn't use it."""


def _split_manifest_mkb(raw: str) -> list[str]:
    out: list[str] = []
    for token in re.split(r"[,;]", raw or ""):
        cleaned = normalize_mkb_code(token)
        if cleaned:
            out.append(cleaned)
    return out


def _load_manifest_mkb_index(path: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    group: dict[str, str] = {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        return exact, group

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = str(row.get("ID") or row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            for code in _split_manifest_mkb(str(row.get("МКБ-10") or row.get("MKB-10") or "")):
                exact.setdefault(code, doc_id)
                group.setdefault(code.split(".", 1)[0], doc_id)
    return exact, group


def _extract_mkb_codes(appointment: dict[str, Any]) -> list[str]:
    found: list[str] = []
    diagnosis = appointment.get("Диагнозы")
    if not isinstance(diagnosis, list):
        return found

    for item in diagnosis:
        if not isinstance(item, dict):
            continue
        code = normalize_mkb_code(str(item.get("КодМКБ", "")))
        if code and code not in found:
            found.append(code)

        detail = str(item.get("Детализация", ""))
        for matched in re.findall(r"\b[A-ZА-Я]\d{2}(?:\.\d+)?\b", detail.upper()):
            candidate = normalize_mkb_code(matched)
            if candidate and candidate not in found:
                found.append(candidate)

    return found


def _resolve_doc_id_by_mkb(mkb_codes: list[str], exact: dict[str, str], group: dict[str, str]) -> str | None:
    for code in mkb_codes:
        doc_id = exact.get(code)
        if doc_id:
            return doc_id
    for code in mkb_codes:
        doc_id = group.get(code.split(".", 1)[0])
        if doc_id:
            return doc_id
    return None


def _load_appointments(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        appointments = payload.get("appointments")
        if isinstance(appointments, list):
            return [item for item in appointments if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Input JSON must be object/list/appointments-wrapper")


def _visit_guid(appointment: dict[str, Any], fallback_idx: int) -> str:
    visit = appointment.get("Прием") if isinstance(appointment.get("Прием"), dict) else {}
    guid = str(visit.get("GUID", "")).strip()
    return guid or f"row_{fallback_idx}"


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    input_path = Path(INPUT_JSON)
    if not input_path.exists() and input_path.name == "parseddata.json":
        input_path = Path("testjson.json")
        log.warning("parseddata.json not found, fallback to %s", input_path)

    appointments = _load_appointments(str(input_path))
    exact, group = _load_manifest_mkb_index(MANIFEST_PATH)

    settings = Settings()
    judge = AppointmentJudge(retrieval=_NoopRetrieval(), settings=settings)

    txt_lines: list[str] = ["# auto-generated; format: doc_id|query"]
    json_rows: list[dict[str, Any]] = []

    processed = 0
    skipped = 0
    query_count = 0

    for idx, appointment in enumerate(appointments, start=1):
        if processed >= MAX_APPOINTMENTS:
            break

        mkb_codes = _extract_mkb_codes(appointment)
        if not mkb_codes:
            skipped += 1
            continue

        doc_id = _resolve_doc_id_by_mkb(mkb_codes, exact, group)
        if not doc_id:
            skipped += 1
            continue

        try:
            queries = judge.build_kg_queries(appointment=appointment, mkb_codes=mkb_codes)
        except Exception as e:
            log.exception("query generation failed for idx=%d: %s", idx, e)
            skipped += 1
            continue

        if not queries:
            skipped += 1
            continue

        guid = _visit_guid(appointment, idx)
        row = {
            "visit_guid": guid,
            "doc_id": doc_id,
            "mkb_codes": mkb_codes,
            "queries": queries,
        }
        json_rows.append(row)

        for q in queries:
            txt_lines.append(f"{doc_id}|{q}")
            query_count += 1

        processed += 1

    Path(OUT_TXT).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    Path(OUT_JSON).write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_file": str(input_path),
        "appointments_total": len(appointments),
        "appointments_used": processed,
        "appointments_skipped": skipped,
        "queries_total": query_count,
        "out_txt": OUT_TXT,
        "out_json": OUT_JSON,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("dataset generated: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
