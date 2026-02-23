from __future__ import annotations

import json
import logging
from typing import Any

from langchain_ollama import ChatOllama

from .config import Settings
from .models import ApiJudgeOutput
from .retrieval import RetrievalService
from .utils import truncate_text

log = logging.getLogger(__name__)

DEFAULT_SCORES_PROMPT = (
    "Ты ассистент внутреннего контроля качества заполнения медицинского приёма в российской клинике. "
    "Оценивай медицинскую корректность заполнения и юридические риски оформления. "
    "Не используй внешние медицинские знания."
)


def normalize_mkb_code(code: str) -> str:
    return code.strip().upper().replace(" ", "")


class AppointmentJudge:
    def __init__(self, retrieval: RetrievalService, settings: Settings, system_prompt: str | None = None):
        self.retrieval = retrieval
        self.settings = settings
        self.system_prompt = system_prompt or DEFAULT_SCORES_PROMPT
        self.chat = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_chat_base_url,
            temperature=0.0,
        )
        self.structured = self.chat.with_structured_output(ApiJudgeOutput, include_raw=True)

    def evaluate_base(self, appointment: dict[str, Any], mkb_codes: list[str]) -> dict[str, Any]:
        prompt = (
            f"{self.system_prompt}\n\n"
            "Этап 1. Базовая проверка заполнения приёма без клинического контекста KG.\n"
            "Проверь полноту, логичность, внутреннюю согласованность и медицинскую адекватность записи.\n"
            "risk_level = юридический риск некорректного заполнения документации (low|medium|high).\n\n"
            f"МКБ в записи: {', '.join(mkb_codes) if mkb_codes else 'не указаны'}\n\n"
            f"JSON приёма:\n{json.dumps(appointment, ensure_ascii=False, indent=2)}"
        )
        return self._invoke_structured(prompt)

    def evaluate_with_kg(
        self,
        doc_id: str,
        appointment: dict[str, Any],
        mkb_codes: list[str],
        context_target: int,
    ) -> tuple[dict[str, Any], list[str]]:
        queries = self.build_kg_queries(appointment=appointment, mkb_codes=mkb_codes)
        log.info("KG evaluation started doc_id=%s queries=%s", doc_id, json.dumps(queries, ensure_ascii=False))
        dedup: dict[str, Any] = {}
        for query in queries:
            query_chunks = self.retrieval.retrieve_context(doc_id=doc_id, query=query)
            log.info("KG query doc_id=%s query=%s retrieved=%d", doc_id, truncate_text(query, 240), len(query_chunks))
            for chunk in query_chunks:
                best = dedup.get(chunk.chunk_id)
                if best is None or chunk.score > best.score:
                    dedup[chunk.chunk_id] = chunk

        top_chunks = sorted(dedup.values(), key=lambda c: c.score, reverse=True)[:context_target]
        for idx, chunk in enumerate(top_chunks[: min(6, len(top_chunks))], start=1):
            snippet = truncate_text(str(getattr(chunk, "chunk_text", "")), 180)
            log.info(
                "KG context #%d doc_id=%s chunk_id=%s section=%s pages=%s-%s type=%s score=%.4f snippet=%s",
                idx,
                doc_id,
                getattr(chunk, "chunk_id", ""),
                getattr(chunk, "section_path", ""),
                getattr(chunk, "page_start", ""),
                getattr(chunk, "page_end", ""),
                getattr(getattr(chunk, "chunk_type", ""), "value", str(getattr(chunk, "chunk_type", ""))),
                float(getattr(chunk, "score", 0.0) or 0.0),
                snippet,
            )
        context_chunks = [self._to_chunk_dict(c) for c in top_chunks]
        prompt = (
            f"{self.system_prompt}\n\n"
            "Этап 2. Проверка с клиническими рекомендациями из knowledge graph.\n"
            "Особенно проверь корректность осмотра, диагноза и рекомендаций относительно контекста.\n"
            "risk_level = юридический риск некорректного заполнения документации (low|medium|high).\n\n"
            f"doc_id рекомендаций: {doc_id}\n"
            f"МКБ в записи: {', '.join(mkb_codes)}\n"
            f"Поисковые запросы:\n{json.dumps(queries, ensure_ascii=False, indent=2)}\n\n"
            f"JSON приёма:\n{json.dumps(appointment, ensure_ascii=False, indent=2)}\n\n"
            f"Контекст KG:\n{json.dumps(context_chunks, ensure_ascii=False, indent=2)}"
        )
        result = self._invoke_structured(prompt)
        log.info("KG evaluation finished doc_id=%s context_chunks=%d", doc_id, len(context_chunks))
        return result, [c["chunk_id"] for c in context_chunks if c.get("chunk_id")]

    def merge_base_and_kg(self, base_scores: dict[str, Any], kg_scores: dict[str, Any]) -> dict[str, Any]:
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

    def build_kg_queries(self, appointment: dict[str, Any], mkb_codes: list[str]) -> list[str]:
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
            qn = q.strip()
            if qn and qn not in uniq:
                uniq.append(qn)
        return uniq[:7]

    def render_human_readable(self, appointment: dict[str, Any]) -> str:
        return json.dumps(appointment, ensure_ascii=False, indent=2)

    def _invoke_structured(self, prompt: str) -> dict[str, Any]:
        resp = self.structured.invoke(prompt)
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

    def _to_chunk_dict(self, chunk: Any) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "section_path": chunk.section_path,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
            "chunk_text": chunk.chunk_text,
        }
