from __future__ import annotations

import logging
import re
from typing import Any

from langchain_ollama import ChatOllama

from .config import Settings
from .models import JudgeResult, VerdictJudgeOutput, VerdictLabel
from .retrieval import RetrievalService
from .storage import WeaviateGraphStore
from .utils import truncate_text

log = logging.getLogger(__name__)


class VerdictJudge:
    def __init__(self, store: WeaviateGraphStore, retrieval: RetrievalService, settings: Settings):
        self.store = store
        self.retrieval = retrieval
        self.settings = settings
        self.chat = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_chat_base_url,
            temperature=0.0,
        )
        self.structured_chat = self.chat.with_structured_output(VerdictJudgeOutput, include_raw=True)

    def evaluate_verdict(self, doc_id: str, verdict_text: str) -> JudgeResult:
        log.info("Verdict evaluation started doc_id=%s verdict_len=%d", doc_id, len(verdict_text))
        queries = [verdict_text] + self._subqueries(verdict_text)
        records = []
        for q in queries[:5]:
            records.extend(self.retrieval.retrieve_context(doc_id=doc_id, query=q))

        dedup: dict[str, Any] = {}
        for r in records:
            if r.doc_id != doc_id:
                continue
            existing = dedup.get(r.chunk_id)
            if existing is None or r.score > existing.score:
                dedup[r.chunk_id] = r

        packed = self.retrieval.pack_context(query=verdict_text, chunk_records=list(dedup.values()), target_n=self.settings.packed_max)

        prompt = self._build_prompt(doc_id=doc_id, verdict_text=verdict_text, chunks=packed)
        response = self.structured_chat.invoke(prompt)
        raw = response.get("raw") if isinstance(response, dict) else None
        parsed = response.get("parsed") if isinstance(response, dict) else None
        parse_error = response.get("parsing_error") if isinstance(response, dict) else None
        raw_text = getattr(raw, "content", str(raw)) if raw is not None else ""
        log.info("LLM raw answer (truncated): %s", truncate_text(str(raw_text), 600))
        if parse_error:
            log.error("Structured output parsing error: %s", parse_error)
        payload_model = parsed or VerdictJudgeOutput(
            verdict=VerdictLabel.INSUFFICIENT_INFO.value,
            explanation="Не удалось получить структурированный ответ от LLM",
            citations=[],
            missing_info=["structured_json_response"],
            recommended_action=None,
        )
        payload = payload_model.model_dump()

        label = payload.get("verdict", VerdictLabel.INSUFFICIENT_INFO.value)
        try:
            verdict = VerdictLabel(label)
        except ValueError:
            verdict = VerdictLabel.INSUFFICIENT_INFO

        result = JudgeResult(
            verdict=verdict,
            explanation=str(payload.get("explanation", "")),
            citations=list(payload.get("citations", [])),
            missing_info=list(payload.get("missing_info", [])),
            recommended_action=payload.get("recommended_action"),
            raw_output=payload,
        )

        self.store.store_verdict_evaluation(
            doc_id=doc_id,
            verdict_text=verdict_text,
            retrieved_chunk_ids=[c.chunk_id for c in packed],
            llm_output=payload,
            model_name=self.settings.ollama_chat_model,
        )
        log.info("Verdict evaluation finished doc_id=%s verdict=%s citations=%d", doc_id, result.verdict.value, len(result.citations))
        return result

    def _subqueries(self, text: str) -> list[str]:
        entities = re.findall(r"\b[А-ЯA-Z][а-яa-zA-Z0-9\-]{2,}\b", text)
        uniq = []
        for e in entities:
            if e not in uniq:
                uniq.append(e)
        return uniq[:4]

    def _build_prompt(self, doc_id: str, verdict_text: str, chunks: list[Any]) -> str:
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"[chunk_id={c.chunk_id}] [section={c.section_path}] [pages={c.page_start}-{c.page_end}] [type={c.chunk_type}]\n{c.chunk_text}"
            )
        context = "\n\n".join(context_parts)
        return (
            "Ты медицинский ассистент для проверки клинического вердикта. Оцени вердикт врача только по приведенным фрагментам. "
            "Если данных недостаточно, верни insufficient_info. Не используй внешние знания.\n\n"
            f"doc_id: {doc_id}\n"
            f"Текст вердикта врача: {verdict_text}\n\n"
            "Контекстные фрагменты:\n"
            f"{context}\n\n"
            "Ответь только JSON-объектом по схеме:\n"
            "{"
            '\"verdict\": \"correct|partially_correct|incorrect|insufficient_info\",'
            '\"explanation\": \"краткое объяснение\",'
            '\"citations\": [{\"chunk_id\":\"...\",\"section_path\":\"...\",\"pages\":\"x-y\"}],'
            '\"missing_info\": [\"...\"],'
            '\"recommended_action\": \"рекомендуемое действие\"'
            "}"
        )
