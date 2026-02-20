from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

from graphrag_weaviate.config import Settings
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.models import ApiJudgeOutput
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore
from graphrag_weaviate.utils import truncate_text

# Edit values below before running.
DOC_ID_OR_MKB_CODE = "I50"
API_INPUT_JSON_PATH = "tools/visit_validator/visit.sample.json"
LOG_FILE = "logs/validate_appointment_pipeline.log"
CONTEXT_TARGET = 10


def _prompt(retrieval_query: str, api_payload: dict[str, Any], context_chunks: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [
            f"[chunk_id={c['chunk_id']}] [section={c['section_path']}] [pages={c['page_start']}-{c['page_end']}] [type={c['chunk_type']}]\n{c['chunk_text']}"
            for c in context_chunks
        ]
    )

    return (
        "Ты ассистент внутреннего контроля качества заполнения медицинского приёма в российской клинике. "
        "Используй ТОЛЬКО предоставленные фрагменты клинических рекомендаций и входной JSON из API. "
        "Не используй внешние медицинские знания.\n\n"
        f"Поисковый запрос для извлечения контекста:\n{retrieval_query}\n\n"
        f"Полный входной JSON из API (оценивать целиком):\n{json.dumps(api_payload, ensure_ascii=False, indent=2)}\n\n"
        "Оцени качество заполнения и верни результат строго по структурированной схеме. "
        "Поле issues: все замечания по секциям, разделитель ';'. "
        "Поле summary: 1-3 строки только значимые выводы для главного врача.\n\n"
        f"Контекстные фрагменты:\n{context_text}"
    )


def _fallback_output() -> dict[str, Any]:
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
        "issues": "Ошибка структурированного ответа LLM; требуется повторная проверка",
        "summary": "Не удалось получить валидный структурированный ответ от LLM.",
    }


def _to_chunk_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
        "chunk_text": chunk.chunk_text,
        "score": chunk.score,
    }


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    settings.packed_max = max(6, min(12, CONTEXT_TARGET))

    api_payload = json.loads(Path(API_INPUT_JSON_PATH).read_text(encoding="utf-8"))
    retrieval_query = str(
        api_payload.get("query")
        or api_payload.get("visit_text")
        or json.dumps(api_payload, ensure_ascii=False)
    )

    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        packed = retrieval.retrieve_context(doc_id=DOC_ID_OR_MKB_CODE, query=retrieval_query)
        context_chunks = [_to_chunk_dict(c) for c in packed]

        llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_chat_base_url, temperature=0.0)
        prompt = _prompt(retrieval_query=retrieval_query, api_payload=api_payload, context_chunks=context_chunks)
        structured_llm = llm.with_structured_output(ApiJudgeOutput, include_raw=True)
        resp = structured_llm.invoke(prompt)

        raw = resp.get("raw") if isinstance(resp, dict) else None
        parsed = resp.get("parsed") if isinstance(resp, dict) else None
        parse_error = resp.get("parsing_error") if isinstance(resp, dict) else None

        raw_text = ""
        if raw is not None:
            raw_text = getattr(raw, "content", str(raw))
        log.info("LLM raw answer (truncated): %s", truncate_text(str(raw_text), 600))

        if parse_error:
            log.error("Structured output parsing error: %s", parse_error)

        if parsed is None:
            evaluation = _fallback_output()
        else:
            evaluation = parsed.model_dump()

        output = {
            "doc_id": DOC_ID_OR_MKB_CODE,
            "query": retrieval_query,
            "api_input": api_payload,
            "retrieved_chunks": context_chunks,
            "evaluation": evaluation,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("Appointment validation pipeline failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
