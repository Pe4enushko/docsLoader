from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama

from engine.config import Settings
from engine.models import ApiJudgeOutput
from engine.retrieval import RetrievalService
from engine.utils import truncate_text

log = logging.getLogger(__name__)

DEFAULT_SCORES_PROMPT = """
Ты — медицинский ИИ-аудитор качества медицинской документации,
работающий для внутреннего контроля медицинской организации
(МИС / ЕГИСЗ / управленческий аудит главного врача).

Твоя задача:
Проанализировать данные медицинского приёма, переданные пользователем,
и оценить заполненность, логическую согласованность и нормативную корректность
медицинской записи в соответствии с требованиями Российской Федерации.

Данные могут быть полуструктурированными, с нумерованными параметрами,
дублирующимися полями, разным порядком разделов и шаблонными формулировками. Они передаются тебе в формате JSON, так что не суди за двоеточия и скобки

Ты НЕ исправляешь и НЕ дополняешь данные.
Ты только анализируешь то, что предоставлено.

---

### НОРМАТИВНАЯ ОСНОВА

Используй как критерии оценки:
- Федеральный закон №323-ФЗ
- Приказ Минздрава РФ №834н
- Приказы Минздрава РФ №514н и №203н
- Требования к ведению данных в ЕГИСЗ
- МКБ-10

---

### ОБЩИЕ ПРАВИЛА АНАЛИЗА

- Самостоятельно сгруппируй входные данные в логические секции
- Учитывай возраст пациента и тип приёма (первичный / повторный / профилактический / контрольный)
- Отсутствие данных ≠ нарушение, если данные не требуются по типу приёма
- Формальные формулировки допустимы, но могут снижать оценку
- Клинические решения врача не оспаривай, если они логически допустимы

---

### КРИТИЧЕСКОЕ ПРАВИЛО ИНТЕРПРЕТАЦИИ

Если приём профилактический:
- отсутствие жалоб при указании «жалоб нет» считается корректным
- отсутствие динамики состояния не является дефектом
- отсутствие лечения допустимо при наличии зафиксированного медицинского решения
- такие отсутствия не снижают оценку секций

---

### СЕКЦИИ ДЛЯ АНАЛИЗА

Используй ровно эти 7 секций:

1. Идентификация и контекст приёма
2. Жалобы и анамнез
3. Объективный осмотр
4. Динамика / клинический контекст
5. Диагноз
6. Медицинские решения и рекомендации
7. Структура и оформление записи

---

### ИНТЕРПРЕТАЦИЯ СЕКЦИЙ

1. Идентификация и контекст приёма
Фиксирует пациента, возраст, дату и тип приёма.
Ошибка: отсутствие возраста, даты или типа приёма.

2. Жалобы и анамнез
Фиксируют основания визита и клинический фон.
Формулировка «жалоб нет» допустима при профилактическом осмотре.

3. Объективный осмотр
Подтверждает факт клинического осмотра.
Общие формулировки допустимы, но снижают детализацию.

4. Динамика / клинический контекст
Используется для повторных и контрольных приёмов.
При профилактическом приёме отсутствие динамики допустимо.

5. Диагноз
Фиксирует медицинское заключение.
Должен соответствовать осмотру и МКБ.

6. Медицинские решения и рекомендации
Отражают итог визита.
Отсутствие лечения допустимо у здорового пациента, но решение должно быть зафиксировано.

7. Структура и оформление записи
Оценивает качество ведения документации и риски аудита.

Ошибки в секциях 3, 5 и 6 имеют повышенный вес при расчёте overall_score.

---

### ШКАЛА ОЦЕНКИ

1 — невозможно анализировать
2 — не заполнено
3 — заполнено частично
4 — заполнено, но с ошибками
5 — заполнено корректно

---

### ИТОГОВАЯ ОЦЕНКА И РИСК

overall_score:
- итоговая оценка с учётом веса ключевых секций

risk_level:
- low → overall_score = 5
- medium → overall_score = 3–4
- high → overall_score = 1–2
или наличие критических ошибок

---

### ФОРМАТ ВЫВОДА (СТРОГО JSON)

Ответ ДОЛЖЕН быть строго в формате JSON:

{
  "overall_score": 1-5,
  "risk_level": "low | medium | high",
  "score_visit_identification": 1-5,
  "score_anamnesis": 1-5,
  "score_inspection": 1-5,
  "score_dynamic": 1-5,
  "score_diagnosis": 1-5,
  "score_recommendations": 1-5,
  "score_structure": 1-5,
  "issues": "Все замечания по каждой секции; разделённые через ';'",
  "summary": "1–3 строки. Только значимые для главного врача выводы."
}

ВАЖНО:
- Проверяй диагноз и медицинские решения строго по данным, которые переданы в prompt.
- Если передан контекст KG, проверяй секции 3, 5 и 6 по нему в первую очередь.
- Никакого текста вне JSON не добавляй.
""".strip()


def normalize_mkb_code(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


class AppointmentJudge:
    def __init__(self, retrieval: RetrievalService, settings: Settings, system_prompt: str | None = None):
        self.retrieval = retrieval
        self.settings = settings
        self.system_prompt = system_prompt or DEFAULT_SCORES_PROMPT
        self.chat = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_chat_base_url,
            temperature=0.0,
            num_ctx=settings.ollama_chat_num_ctx,
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
        return self._invoke_structured(prompt, request_name="base_evaluation")

    def evaluate_with_kg(
        self,
        doc_id: str,
        doc_title: str | None,
        appointment: dict[str, Any],
        mkb_codes: list[str],
        context_target: int,
    ) -> tuple[dict[str, Any], list[str]]:
        queries = self.build_kg_queries(appointment=appointment, mkb_codes=mkb_codes)
        log.info(
            "KG evaluation started doc_id=%s doc_title=%s queries=%s",
            doc_id,
            doc_title or "<missing>",
            json.dumps(queries, ensure_ascii=False),
        )
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
            "Ссылаясь на рекоммендации в issues/summary, указывай только читаемые метаданные: "
            "раздел (section_path), страницы (page_start-page_end) и тип фрагмента (chunk_type). Обязательно переводи метаданные в человекочитаемый вид, не используй технические названия полей. "
            "Не используй chunk_id в тексте ответа.\n "
            "Всегда указывай ссылки на контекст (он же клинические рекоммендации). "
            "Обязательно переводи метаданные в человекочитаемый вид на русском языке, не используй технические названия полей.\n\n"
            f"doc_id рекомендаций: {doc_id}\n"
            f"Наименование документа: {doc_title or 'не указано'}\n"
            f"МКБ в записи: {', '.join(mkb_codes)}\n"
            f"Поисковые запросы:\n{json.dumps(queries, ensure_ascii=False, indent=2)}\n\n"
            f"JSON приёма:\n{json.dumps(appointment, ensure_ascii=False, indent=2)}\n\n"
            f"Контекст:\n{json.dumps(context_chunks, ensure_ascii=False, indent=2)}"
        )
        result = self._invoke_structured(prompt, request_name="kg_evaluation")
        log.info("KG evaluation finished doc_id=%s context_chunks=%d", doc_id, len(context_chunks))
        result["summary"] = f"{result.get('summary', '').strip()}\nДокументы: {doc_id} — {doc_title or 'не указано'}".strip()
        return result, [getattr(c, "chunk_id", "") for c in top_chunks if getattr(c, "chunk_id", "")]

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
        lines: list[str] = []

        for key, value in appointment.items():
            if key == "ДанныеОсмотра" and isinstance(value, list):
                lines.append("ДанныеОсмотра:")
                for item in value:
                    if not isinstance(item, dict):
                        lines.append(f"- {self._format_scalar(item)}")
                        continue
                    param = str(item.get("Параметр", "")).strip()
                    val = str(item.get("Значение", "")).strip()
                    if param or val:
                        lines.append(f"{param}: {val}".strip(": ").strip())
                    else:
                        lines.append("-")
                lines.append("")
                continue

            lines.extend(self._format_block(key, value))
            lines.append("")

        return "\n".join(line for line in lines).strip()

    def _format_block(self, key: str, value: Any) -> list[str]:
        out: list[str] = [f"{key}:"]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out.extend(self._format_kv(sub_key, sub_value))
            return out
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        out.extend(self._format_kv(sub_key, sub_value))
                    out.append("")
                else:
                    out.append(f"- {self._format_scalar(item)}")
            if out and out[-1] == "":
                out.pop()
            return out
        out.append(self._format_scalar(value))
        return out

    def _format_kv(self, key: str, value: Any) -> list[str]:
        if isinstance(value, dict):
            out: list[str] = [f"{key}:"]
            for sub_key, sub_value in value.items():
                out.extend(self._format_kv(sub_key, sub_value))
            return out
        if isinstance(value, list):
            out = [f"{key}:"]
            for item in value:
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        out.extend(self._format_kv(sub_key, sub_value))
                    out.append("")
                else:
                    out.append(f"- {self._format_scalar(item)}")
            if out and out[-1] == "":
                out.pop()
            return out
        return [f"{key}: {self._format_scalar(value)}"]

    def _format_scalar(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    def _invoke_structured(self, prompt: str, request_name: str) -> dict[str, Any]:
        log.info(
            "LLM request started type=%s prompt_words=%d prompt_snippet=%s",
            request_name,
            _word_count(prompt),
            truncate_text(prompt, 500),
        )
        resp = self.structured.invoke(prompt)
        raw = resp.get("raw") if isinstance(resp, dict) else None
        parsed = resp.get("parsed") if isinstance(resp, dict) else None
        parse_error = resp.get("parsing_error") if isinstance(resp, dict) else None

        raw_text = getattr(raw, "content", str(raw)) if raw is not None else ""
        if raw_text:
            log.info("LLM raw answer type=%s: %s", request_name, str(raw_text))
        if parse_error:
            log.error("Structured output parsing error type=%s: %s", request_name, parse_error)

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
        section = getattr(chunk, "section_path", "")
        page_start = getattr(chunk, "page_start", "")
        page_end = getattr(chunk, "page_end", "")
        chunk_type = getattr(getattr(chunk, "chunk_type", ""), "value", str(getattr(chunk, "chunk_type", "")))
        return {
            "source_ref": f"section={section}; pages={page_start}-{page_end}; type={chunk_type}",
            "section_path": section,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_type": chunk_type,
            "chunk_text": getattr(chunk, "chunk_text", ""),
        }
