from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphrag_weaviate.config import Settings

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:  # type: ignore[redef]
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate medical visit verdict/diagnosis/recommendations against doc-scoped knowledge graph"
    )
    code_group = parser.add_mutually_exclusive_group(required=True)
    code_group.add_argument("--doc_id", help="Guideline document ID for strict filtering")
    code_group.add_argument("--mkb_code", help="MKB code used as document ID")
    parser.add_argument("--visit_json", default=None, help="Path to JSON file with visit data")
    parser.add_argument("--query", default=None, help="Free-text visit data")
    parser.add_argument("--top_k", type=int, default=10, help="Packed context size (6-12 recommended)")
    parser.add_argument("--section_prefix", default=None)
    parser.add_argument("--chunk_types", default=None, help="Comma-separated chunk type allowlist")
    parser.add_argument("--page_start", type=int, default=None)
    parser.add_argument("--page_end", type=int, default=None)
    return parser


def _load_visit_data(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.visit_json:
        payload = json.loads(Path(args.visit_json).read_text(encoding="utf-8"))
        query = payload.get("query") or payload.get("visit_text") or json.dumps(payload, ensure_ascii=False)
        return str(query), payload
    if args.query:
        return args.query, {"query": args.query}
    raise ValueError("Provide --visit_json or --query")


def _prompt(query: str, visit_data: dict[str, Any], context_chunks: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [
            f"[chunk_id={c['chunk_id']}] [section={c['section_path']}] [pages={c['page_start']}-{c['page_end']}] [type={c['chunk_type']}]\n{c['chunk_text']}"
            for c in context_chunks
        ]
    )

    return (
        "Ты ассистент для клинической валидации. Используй ТОЛЬКО предоставленные фрагменты из одного документа рекомендаций. "
        "Не используй внешние медицинские знания.\n\n"
        f"Запрос по визиту:\n{query}\n\n"
        f"Структурированные данные визита (если есть):\n{json.dumps(visit_data, ensure_ascii=False, indent=2)}\n\n"
        "Верни ТОЛЬКО валидный JSON по схеме:\n"
        "{"
        '"overall": "supported|partially_supported|not_supported|insufficient_info",'
        '"diagnosis_check": {"status": "supported|partially_supported|not_supported|insufficient_info", "reason": "...", "citations": [{"chunk_id":"...","section_path":"...","pages":"x-y"}]},'
        '"recommendations_check": {"status": "supported|partially_supported|not_supported|insufficient_info", "reason": "...", "citations": [{"chunk_id":"...","section_path":"...","pages":"x-y"}]},'
        '"verdict_check": {"status": "supported|partially_supported|not_supported|insufficient_info", "reason": "...", "citations": [{"chunk_id":"...","section_path":"...","pages":"x-y"}]},'
        '"missing_info": ["..."],'
        '"action_items": ["..."]'
        "}\n\n"
        f"Контекстные фрагменты:\n{context_text}"
    )


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "overall": "insufficient_info",
            "diagnosis_check": {
                "status": "insufficient_info",
                "reason": "LLM returned invalid JSON",
                "citations": [],
            },
            "recommendations_check": {
                "status": "insufficient_info",
                "reason": "LLM returned invalid JSON",
                "citations": [],
            },
            "verdict_check": {
                "status": "insufficient_info",
                "reason": "LLM returned invalid JSON",
                "citations": [],
            },
            "missing_info": ["valid_json_output"],
            "action_items": [],
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
    load_dotenv()
    args = _build_parser().parse_args()
    from langchain_ollama import ChatOllama
    from graphrag_weaviate.retrieval import RetrievalService
    from graphrag_weaviate.storage import WeaviateGraphStore

    doc_id = args.doc_id or args.mkb_code
    settings = Settings()
    settings.packed_max = max(6, min(12, args.top_k))

    query, visit_data = _load_visit_data(args)

    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        page_range = None
        if args.page_start is not None and args.page_end is not None:
            page_range = (args.page_start, args.page_end)
        chunk_types = [x.strip() for x in args.chunk_types.split(",")] if args.chunk_types else None

        packed = retrieval.retrieve_context(
            doc_id=doc_id,
            query=query,
            section_prefix=args.section_prefix,
            chunk_type_allowlist=chunk_types,
            page_range=page_range,
        )
        context_chunks = [_to_chunk_dict(c) for c in packed]

        llm = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_chat_base_url,
            temperature=0.0,
        )
        prompt = _prompt(query=query, visit_data=visit_data, context_chunks=context_chunks)
        resp = llm.invoke(prompt)
        evaluation = _parse_json(resp.content)

        print(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "query": query,
                    "retrieved_chunks": context_chunks,
                    "evaluation": evaluation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
