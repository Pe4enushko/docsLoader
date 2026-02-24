from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.appointments import (
    extract_visit_date_raw,
    extract_visit_guid as shared_extract_visit_guid,
)
from engine.config import Settings
from engine.integrations.one_c import OneCClient
from engine.llm import AppointmentJudge, normalize_mkb_code
from engine.logging_utils import setup_logging
from engine.models import MedKardRow
from engine.retrieval import RetrievalService
from engine.storage import WeaviateGraphStore
from engine.postgres import connect_postgres, ensure_medkard_table, is_visit_processed, upsert_medkard_row

MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.csv")
LOG_FILE = "logs/docsToGraphRAG.log"
CONTEXT_TARGET = int(os.getenv("CONTEXT_TARGET", "8"))


def fetch_appointments_from_1c() -> list[dict[str, Any]]:
    client = OneCClient.from_env()
    return client.fetch_appointments_for_today()


def split_manifest_mkb(raw: str) -> list[str]:
    out = []
    for token in re.split(r"[,;]", raw or ""):
        cleaned = normalize_mkb_code(token)
        if cleaned:
            out.append(cleaned)
    return out


def load_manifest_mkb_index(path: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    group: dict[str, str] = {}
    titles_by_doc_id: dict[str, str] = {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        return exact, group, titles_by_doc_id
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = str(row.get("ID") or row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            title = str(row.get("Наименование") or row.get("title") or "").strip()
            if title:
                titles_by_doc_id.setdefault(doc_id, title)
            for code in split_manifest_mkb(str(row.get("МКБ-10") or row.get("MKB-10") or "")):
                exact.setdefault(code, doc_id)
                group.setdefault(code.split(".", 1)[0], doc_id)
    return exact, group, titles_by_doc_id


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
    manifest_titles: dict[str, str],
) -> MedKardRow:
    mkb_codes = extract_mkb_codes(appointment)
    doc_id = resolve_doc_id_by_mkb(mkb_codes, manifest_exact, manifest_group)
    if mkb_codes and doc_id:
        doc_title = manifest_titles.get(doc_id)
        kg_scores, _ = judge.evaluate_with_kg(
            doc_id=doc_id,
            doc_title=doc_title,
            appointment=appointment,
            mkb_codes=mkb_codes,
            context_target=CONTEXT_TARGET,
        )
        final_scores = kg_scores
    else:
        final_scores = judge.evaluate_base(appointment=appointment, mkb_codes=mkb_codes)
        if mkb_codes and not doc_id:
            note = "МКБ найден, но не сопоставлен с manifest.csv; KG-проверка пропущена"
            existing = str(final_scores.get("issues", "")).strip()
            final_scores["issues"] = f"{existing}; {note}" if existing else note

    human_readable = judge.render_human_readable(appointment=appointment)

    visit_guid = shared_extract_visit_guid(appointment)
    if not visit_guid:
        raise ValueError("Missing Прием.GUID for appointment")

    return MedKardRow(
        visit_guid_1c=visit_guid,
        score_visit_identification=int(final_scores["score_visit_identification"]),
        score_anamnes=int(final_scores["score_anamnesis"]),
        score_inspection=int(final_scores["score_inspection"]),
        score_dynamic=int(final_scores["score_dynamic"]),
        score_diagnosis=int(final_scores["score_diagnosis"]),
        score_recommendations=int(final_scores["score_recommendations"]),
        score_structure=int(final_scores["score_structure"]),
        issues=str(final_scores.get("issues", "")).strip() or None,
        summary=str(final_scores.get("summary", "")).strip() or None,
        score_overall=int(final_scores["overall_score"]),
        risk_level=str(final_scores["risk_level"]),
        inspection_data=json.dumps(appointment.get("ДанныеОсмотра"), ensure_ascii=False)
        if appointment.get("ДанныеОсмотра") is not None
        else None,
        diagnosis_data=json.dumps(appointment.get("Диагнозы"), ensure_ascii=False)
        if appointment.get("Диагнозы") is not None
        else None,
        services_data=json.dumps(appointment.get("Услуги"), ensure_ascii=False)
        if appointment.get("Услуги") is not None
        else None,
        visit_date=parse_visit_date(extract_visit_date_raw(appointment)),
        human_readable=human_readable or None,
        patient=json.dumps(appointment.get("Пациент"), ensure_ascii=False)
        if appointment.get("Пациент") is not None
        else None,
    )


def extract_visit_guid(appointment: dict[str, Any]) -> str:
    return shared_extract_visit_guid(appointment)


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = AppointmentJudge(retrieval=retrieval, settings=settings)

        manifest_exact, manifest_group, manifest_titles = load_manifest_mkb_index(MANIFEST_PATH)
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
                    manifest_titles=manifest_titles,
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
