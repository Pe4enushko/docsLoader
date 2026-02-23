from __future__ import annotations

import base64
import csv
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from graphrag_weaviate.config import Settings
from graphrag_weaviate.llm import AppointmentJudge, normalize_mkb_code
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore
from medkard_postgres import connect_postgres, ensure_medkard_table, is_visit_processed, upsert_medkard_row

MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.csv")
ONE_C_APPOINTMENTS_URL = os.getenv("ONE_C_APPOINTMENTS_URL", "<ONE_C_APPOINTMENTS_URL>")
ONE_C_LOGIN = os.getenv("ONE_C_LOGIN") or ""
ONE_C_PASSWORD = os.getenv("ONE_C_PASSWORD") or ""
ONE_C_TIMEOUT_SECONDS = float(os.getenv("ONE_C_TIMEOUT_SECONDS", "15"))
LOG_FILE = "logs/docsToGraphRAG.log"
CONTEXT_TARGET = int(os.getenv("CONTEXT_TARGET", "8"))


def fetch_appointments_from_1c() -> list[dict[str, Any]]:
    if not ONE_C_APPOINTMENTS_URL or ONE_C_APPOINTMENTS_URL.startswith("<"):
        raise ValueError("Set real ONE_C_APPOINTMENTS_URL in environment")
    if not ONE_C_LOGIN or not ONE_C_PASSWORD:
        raise ValueError("ONE_C_LOGIN and ONE_C_PASSWORD must be set")

    basic_token = base64.b64encode(f"{ONE_C_LOGIN}:{ONE_C_PASSWORD}".encode("utf-8")).decode("ascii")
    current_day = datetime.now().strftime("%d.%m.%Y")
    query_params = urllib.parse.urlencode({"datebegin": current_day, "dateend": current_day})
    separator = "&" if "?" in ONE_C_APPOINTMENTS_URL else "?"
    request_url = f"{ONE_C_APPOINTMENTS_URL}{separator}{query_params}"

    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json", "Authorization": f"Basic {basic_token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=ONE_C_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch appointments from 1C: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("HTTP response must be JSON object with 'appointments'")
    appointments = payload.get("appointments")
    if not isinstance(appointments, list):
        raise ValueError("HTTP response must contain 'appointments' array")
    return [item for item in appointments if isinstance(item, dict)]


def split_manifest_mkb(raw: str) -> list[str]:
    out = []
    for token in re.split(r"[,;]", raw or ""):
        cleaned = normalize_mkb_code(token)
        if cleaned:
            out.append(cleaned)
    return out


def load_manifest_mkb_index(path: str) -> tuple[dict[str, str], dict[str, str]]:
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
            for code in split_manifest_mkb(str(row.get("МКБ-10") or row.get("MKB-10") or "")):
                exact.setdefault(code, doc_id)
                group.setdefault(code.split(".", 1)[0], doc_id)
    return exact, group


def extract_mkb_codes(appointment: dict[str, Any]) -> list[str]:
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


def resolve_doc_id_by_mkb(mkb_codes: list[str], exact: dict[str, str], group: dict[str, str]) -> str | None:
    for code in mkb_codes:
        doc_id = exact.get(code)
        if doc_id:
            return doc_id
    for code in mkb_codes:
        doc_id = group.get(code.split(".", 1)[0])
        if doc_id:
            return doc_id
    return None


def parse_visit_date(raw_date: Any) -> str | None:
    value = str(raw_date or "").strip()
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def build_row_for_medkard(
    judge: AppointmentJudge,
    appointment: dict[str, Any],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
) -> tuple:
    mkb_codes = extract_mkb_codes(appointment)
    base_scores = judge.evaluate_base(appointment=appointment, mkb_codes=mkb_codes)

    final_scores = base_scores
    doc_id = resolve_doc_id_by_mkb(mkb_codes, manifest_exact, manifest_group)
    if mkb_codes and doc_id:
        kg_scores, _ = judge.evaluate_with_kg(
            doc_id=doc_id,
            appointment=appointment,
            mkb_codes=mkb_codes,
            context_target=CONTEXT_TARGET,
        )
        final_scores = judge.merge_base_and_kg(base_scores=base_scores, kg_scores=kg_scores)
    elif mkb_codes and not doc_id:
        note = "МКБ найден, но не сопоставлен с manifest.csv; KG-проверка пропущена"
        existing = str(base_scores.get("issues", "")).strip()
        final_scores["issues"] = f"{existing}; {note}" if existing else note

    human_readable = judge.render_human_readable(appointment=appointment)

    visit = appointment.get("Прием") if isinstance(appointment.get("Прием"), dict) else {}
    visit_guid = str(visit.get("GUID", "")).strip()
    if not visit_guid:
        raise ValueError("Missing Прием.GUID for appointment")

    return (
        visit_guid,
        int(final_scores["score_visit_identification"]),
        int(final_scores["score_anamnesis"]),
        int(final_scores["score_inspection"]),
        int(final_scores["score_dynamic"]),
        int(final_scores["score_diagnosis"]),
        int(final_scores["score_recommendations"]),
        int(final_scores["score_structure"]),
        str(final_scores.get("issues", "")).strip() or None,
        str(final_scores.get("summary", "")).strip() or None,
        int(final_scores["overall_score"]),
        str(final_scores["risk_level"]),
        json.dumps(appointment.get("ДанныеОсмотра"), ensure_ascii=False) if appointment.get("ДанныеОсмотра") is not None else None,
        json.dumps(appointment.get("Диагнозы"), ensure_ascii=False) if appointment.get("Диагнозы") is not None else None,
        json.dumps(appointment.get("Услуги"), ensure_ascii=False) if appointment.get("Услуги") is not None else None,
        parse_visit_date(visit.get("DATE")),
        human_readable or None,
        json.dumps(appointment.get("Пациент"), ensure_ascii=False) if appointment.get("Пациент") is not None else None,
    )


def extract_visit_guid(appointment: dict[str, Any]) -> str:
    visit = appointment.get("Прием") if isinstance(appointment.get("Прием"), dict) else {}
    return str(visit.get("GUID", "")).strip()


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = AppointmentJudge(retrieval=retrieval, settings=settings)

        manifest_exact, manifest_group = load_manifest_mkb_index(MANIFEST_PATH)
        appointments = fetch_appointments_from_1c()

        with connect_postgres() as conn:
            ensure_medkard_table(conn)
            processed = 0
            skipped = 0
            for item in appointments:
                visit_guid = extract_visit_guid(item)
                if not visit_guid:
                    raise ValueError("Missing Прием.GUID for appointment")
                if is_visit_processed(conn, visit_guid):
                    skipped += 1
                    continue

                row = build_row_for_medkard(
                    judge=judge,
                    appointment=item,
                    manifest_exact=manifest_exact,
                    manifest_group=manifest_group,
                )
                upsert_medkard_row(conn, row)
                conn.commit()
                processed += 1

        print(json.dumps({"processed": processed, "skipped": skipped}, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("MedKard evaluation pipeline failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
