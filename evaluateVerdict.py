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

from langchain_ollama import ChatOllama

from graphrag_weaviate.config import Settings
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.models import ApiJudgeOutput
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore
from graphrag_weaviate.utils import truncate_text
from medkard_postgres import connect_postgres, ensure_medkard_table, upsert_medkard_rows

MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.csv")
ONE_C_APPOINTMENTS_URL = os.getenv("ONE_C_APPOINTMENTS_URL", "<ONE_C_APPOINTMENTS_URL>")
ONE_C_LOGIN = os.getenv("ONE_C_LOGIN") or os.getenv("ONE_C_USERNAME") or os.getenv("LOGIN") or ""
ONE_C_PASSWORD = os.getenv("ONE_C_PASSWORD") or os.getenv("PASSWORD") or ""
ONE_C_TIMEOUT_SECONDS = float(os.getenv("ONE_C_TIMEOUT_SECONDS", "15"))
LOG_FILE = "logs/docsToGraphRAG.log"
CONTEXT_TARGET = int(os.getenv("CONTEXT_TARGET", "10"))
SCORES_SYSTEM_PROMPT = os.getenv(
    "SCORES_SYSTEM_PROMPT",
    (
        "Ты ассистент внутреннего контроля качества заполнения медицинского приёма в российской клинике. "
        "Оценивай медицинскую корректность заполнения и юридические риски оформления. "
        "Не используй внешние медицинские знания."
    ),
)


def fetch_appointments_from_1c() -> list[dict[str, Any]]:
    if not ONE_C_APPOINTMENTS_URL or ONE_C_APPOINTMENTS_URL.startswith("<"):
        raise ValueError("Set real ONE_C_APPOINTMENTS_URL in environment")
    if not ONE_C_LOGIN or not ONE_C_PASSWORD:
        raise ValueError("ONE_C_LOGIN and ONE_C_PASSWORD (or LOGIN/PASSWORD) must be set")

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


def normalize_mkb_code(code: str) -> str:
    return code.strip().upper().replace(" ", "")


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


def resolve_doc_id_by_mkb(mkb_codes: list[str], exact: dict[str, str], group: dict[str, str]) -> tuple[str | None, str | None]:
    for code in mkb_codes:
        doc_id = exact.get(code)
        if doc_id:
            return doc_id, code
    for code in mkb_codes:
        doc_id = group.get(code.split(".", 1)[0])
        if doc_id:
            return doc_id, code
    return None, None


def _invoke_structured(llm: ChatOllama, prompt: str, log: logging.Logger) -> dict[str, Any]:
    structured_llm = llm.with_structured_output(ApiJudgeOutput, include_raw=True)
    resp = structured_llm.invoke(prompt)
    raw = resp.get("raw") if isinstance(resp, dict) else None
    parsed = resp.get("parsed") if isinstance(resp, dict) else None
    parse_error = resp.get("parsing_error") if isinstance(resp, dict) else None

    raw_text = getattr(raw, "content", str(raw)) if raw is not None else ""
    if raw_text:
        log.info("LLM raw answer (truncated): %s", truncate_text(str(raw_text), 500))
    if parse_error:
        log.error("Structured output parsing error: %s", parse_error)
    if parsed is None:
        return {
            "overall_score": 1,
            "risk_level": "high",
            "score_visit_identification": 1,
            "score_anamnesis": 1,
            "score_inspection": 1,
            "score_dynamic": 1,
            "score_diagnosis": 1,
            "score_recommendations": 1,
            "score_structure": 1,
            "issues": "Ошибка структурированного ответа LLM",
            "summary": "Не удалось получить валидный структурированный ответ от LLM",
        }
    return parsed.model_dump()


def build_base_prompt(appointment: dict[str, Any], mkb_codes: list[str]) -> str:
    return (
        f"{SCORES_SYSTEM_PROMPT}\n\n"
        "Этап 1. Базовая проверка заполнения приёма без клинического контекста KG.\n"
        "Проверь полноту, логичность, внутреннюю согласованность и медицинскую адекватность записи.\n"
        "risk_level = юридический риск некорректного заполнения документации (low|medium|high).\n\n"
        f"МКБ в записи: {', '.join(mkb_codes) if mkb_codes else 'не указаны'}\n\n"
        f"JSON приёма:\n{json.dumps(appointment, ensure_ascii=False, indent=2)}"
    )


def build_kg_queries(appointment: dict[str, Any], mkb_codes: list[str]) -> list[str]:
    queries: list[str] = []
    diagnosis = appointment.get("Диагнозы")
    if isinstance(diagnosis, list):
        for item in diagnosis:
            if not isinstance(item, dict):
                continue
            combined = " ".join(
                part
                for part in [
                    normalize_mkb_code(str(item.get("КодМКБ", ""))),
                    str(item.get("НаименованиеМКБ", "")).strip(),
                    str(item.get("Детализация", "")).strip(),
                ]
                if part
            ).strip()
            if combined:
                queries.append(combined)
    inspection = appointment.get("ДанныеОсмотра")
    if isinstance(inspection, list):
        for item in inspection:
            if not isinstance(item, dict):
                continue
            param = str(item.get("Параметр", "")).lower()
            value = str(item.get("Значение", "")).strip()
            if not value:
                continue
            if "рекомендац" in param or "осмотр" in param or "динамик" in param or "анамн" in param:
                queries.append(value[:500])
    if mkb_codes:
        queries.append(" ".join(mkb_codes))
    if not queries:
        queries.append(json.dumps(appointment, ensure_ascii=False)[:1000])
    uniq: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in uniq:
            uniq.append(q)
    return uniq[:7]


def _to_chunk_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
        "chunk_text": chunk.chunk_text,
    }


def build_kg_prompt(
    appointment: dict[str, Any],
    mkb_codes: list[str],
    doc_id: str,
    queries: list[str],
    context_chunks: list[dict[str, Any]],
) -> str:
    return (
        f"{SCORES_SYSTEM_PROMPT}\n\n"
        "Этап 2. Проверка с клиническими рекомендациями из knowledge graph.\n"
        "Особенно проверь корректность осмотра, диагноза и рекомендаций относительно контекста.\n"
        "risk_level = юридический риск некорректного заполнения документации (low|medium|high).\n\n"
        f"doc_id рекомендаций: {doc_id}\n"
        f"МКБ в записи: {', '.join(mkb_codes)}\n"
        f"Поисковые запросы:\n{json.dumps(queries, ensure_ascii=False, indent=2)}\n\n"
        f"JSON приёма:\n{json.dumps(appointment, ensure_ascii=False, indent=2)}\n\n"
        f"Контекст KG:\n{json.dumps(context_chunks, ensure_ascii=False, indent=2)}"
    )


def merge_scores(base_scores: dict[str, Any], kg_scores: dict[str, Any]) -> dict[str, Any]:
    out = dict(base_scores)
    out["score_inspection"] = int(kg_scores["score_inspection"])
    out["score_diagnosis"] = int(kg_scores["score_diagnosis"])
    out["score_recommendations"] = int(kg_scores["score_recommendations"])

    total = (
        int(out["score_visit_identification"])
        + int(out["score_anamnesis"])
        + int(out["score_inspection"])
        + int(out["score_dynamic"])
        + int(out["score_diagnosis"])
        + int(out["score_recommendations"])
        + int(out["score_structure"])
    )
    out["overall_score"] = max(1, min(5, round(total / 7)))

    rank = {"low": 1, "medium": 2, "high": 3}
    base_risk = str(base_scores.get("risk_level", "high"))
    kg_risk = str(kg_scores.get("risk_level", "high"))
    out["risk_level"] = base_risk if rank.get(base_risk, 3) >= rank.get(kg_risk, 3) else kg_risk

    issues = []
    base_issues = str(base_scores.get("issues", "")).strip()
    kg_issues = str(kg_scores.get("issues", "")).strip()
    if base_issues:
        issues.append(base_issues)
    if kg_issues:
        issues.append(kg_issues)
    out["issues"] = "; ".join(issues) if issues else None

    kg_summary = str(kg_scores.get("summary", "")).strip()
    out["summary"] = kg_summary or str(base_scores.get("summary", "")).strip()
    return out


def generate_human_readable(appointment: dict[str, Any]) -> str:
    return json.dumps(appointment, ensure_ascii=False, indent=2)


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
    llm: ChatOllama,
    retrieval: RetrievalService,
    appointment: dict[str, Any],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
    log: logging.Logger,
) -> tuple:
    mkb_codes = extract_mkb_codes(appointment)
    base_scores = _invoke_structured(llm=llm, prompt=build_base_prompt(appointment, mkb_codes), log=log)

    doc_id, _ = resolve_doc_id_by_mkb(mkb_codes, manifest_exact, manifest_group)
    final_scores = base_scores
    if mkb_codes and doc_id:
        queries = build_kg_queries(appointment, mkb_codes)
        dedup: dict[str, Any] = {}
        for q in queries:
            for chunk in retrieval.retrieve_context(doc_id=doc_id, query=q):
                best = dedup.get(chunk.chunk_id)
                if best is None or chunk.score > best.score:
                    dedup[chunk.chunk_id] = chunk
        top_chunks = sorted(dedup.values(), key=lambda c: c.score, reverse=True)[:CONTEXT_TARGET]
        context_chunks = [_to_chunk_dict(c) for c in top_chunks]
        kg_prompt = build_kg_prompt(appointment, mkb_codes, doc_id, queries, context_chunks)
        kg_scores = _invoke_structured(llm=llm, prompt=kg_prompt, log=log)
        final_scores = merge_scores(base_scores, kg_scores)
    elif mkb_codes and not doc_id:
        note = "МКБ найден, но не сопоставлен с manifest.csv; KG-проверка пропущена"
        existing = str(base_scores.get("issues", "")).strip()
        final_scores["issues"] = f"{existing}; {note}" if existing else note

    human_readable = generate_human_readable(appointment=appointment)

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


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_chat_base_url, temperature=0.0)
        retrieval = RetrievalService(store, settings)

        manifest_exact, manifest_group = load_manifest_mkb_index(MANIFEST_PATH)
        appointments = fetch_appointments_from_1c()

        rows = []
        for item in appointments:
            rows.append(
                build_row_for_medkard(
                    llm=llm,
                    retrieval=retrieval,
                    appointment=item,
                    manifest_exact=manifest_exact,
                    manifest_group=manifest_group,
                    log=log,
                )
            )

        with connect_postgres() as conn:
            ensure_medkard_table(conn)
            upsert_medkard_rows(conn, rows)
            conn.commit()

        print(json.dumps({"processed": len(rows)}, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("MedKard evaluation pipeline failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
