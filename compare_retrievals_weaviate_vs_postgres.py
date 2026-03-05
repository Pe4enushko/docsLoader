from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path

from langchain_ollama import ChatOllama

from engine.config import Settings
from engine.graphrag import PgvectorAgeAdapter, RetrievalService, WeaviateKnowledgeGraphAdapter, build_graphrag_postgres_dsn
from engine.logging_utils import setup_logging
from engine.models import ChunkRecord
from engine.weaviate import WeaviateGraphStore

DATASET_FILE = os.getenv("RETRIEVAL_DATASET_FILE", "retrieval_dataset.generated.json")
RESULTS_FILE = os.getenv("RETRIEVAL_COMPARE_RESULTS", "retrieval_compare_results.generated.csv")
LOG_FILE = "logs/compare_retrievals_weaviate_vs_postgres.log"


def load_generated_dataset(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        visit_guid = str(item.get("visit_guid", "")).strip()
        mkb_codes = item.get("mkb_codes")
        queries = item.get("queries")
        if not doc_id or not isinstance(queries, list) or not queries:
            continue
        cleaned_queries = [str(q).strip() for q in queries if str(q).strip()]
        if not cleaned_queries:
            continue
        out.append(
            {
                "visit_guid": visit_guid or None,
                "doc_id": doc_id,
                "mkb_codes": mkb_codes if isinstance(mkb_codes, list) else [],
                "queries": cleaned_queries,
            }
        )
    return out


def to_context(chunks: list[ChunkRecord], top_n: int = 6) -> list[dict]:
    rows: list[dict] = []
    for c in chunks[:top_n]:
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "pages": f"{c.page_start}-{c.page_end}",
                "chunk_type": getattr(c.chunk_type, "value", str(c.chunk_type)),
                "score": round(float(c.score), 4),
                "text": c.chunk_text[:700],
            }
        )
    return rows


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def clamp_percent(value: object) -> int:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(as_float))))


def judge_compare(chat: ChatOllama, doc_id: str, query: str, weav_ctx: list[dict], pg_ctx: list[dict]) -> str:
    prompt = (
        "Ты судья качества retrieval-контекста для медицинского RAG.\n"
        "Оцени качество каждого retrieval-контекста по релевантности запросу.\n"
        "Верни строго JSON:\n"
        '{ "postgres_score": 0..100, "weaviate_score": 0..100, "reason": "кратко" }\n'
        "Где 0 - нерелевантно, 100 - максимально релевантно.\n\n"
        f"doc_id: {doc_id}\n"
        f"query: {query}\n\n"
        f"weaviate_context:\n{json.dumps(weav_ctx, ensure_ascii=False, indent=2)}\n\n"
        f"postgres_context:\n{json.dumps(pg_ctx, ensure_ascii=False, indent=2)}\n"
    )
    return chat.invoke(prompt).content


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)
    settings = Settings()
    dataset_rows = load_generated_dataset(DATASET_FILE)

    if PgvectorAgeAdapter is None:
        raise RuntimeError("Postgres adapter is unavailable. Install psycopg2-binary.")

    weav_store = WeaviateGraphStore(settings)
    pg_adapter = PgvectorAgeAdapter(dsn=build_graphrag_postgres_dsn(), settings=settings)
    chat = ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_chat_base_url,
        temperature=0.0,
        num_ctx=settings.ollama_chat_num_ctx,
    )

    try:
        weav_retrieval = RetrievalService(WeaviateKnowledgeGraphAdapter(weav_store), settings)
        pg_retrieval = RetrievalService(pg_adapter, settings)

        fieldnames = [
            "doc_id",
            "query",
            "postgres_score",
            "weaviate_score",
            "postgres_speed",
            "weaviate_speed",
        ]
        results_path = Path(RESULTS_FILE)
        should_write_header = (not results_path.exists()) or results_path.stat().st_size == 0
        queries_total = 0

        with results_path.open("a", encoding="utf-8", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter="|")
            if should_write_header:
                writer.writeheader()

            idx = 0
            for row in dataset_rows:
                doc_id = row["doc_id"]

                for query in row["queries"]:
                    idx += 1
                    queries_total += 1

                    weav_start = time.time()
                    weav_chunks = weav_retrieval.retrieve_context(doc_id=doc_id, query=query)
                    weav_end = time.time()
                    weav_speed = round(weav_end - weav_start, 6)

                    pg_start = time.time()
                    pg_chunks = pg_retrieval.retrieve_context(doc_id=doc_id, query=query)
                    pg_end = time.time()
                    pg_speed = round(pg_end - pg_start, 6)

                    weav_ctx = to_context(weav_chunks)
                    pg_ctx = to_context(pg_chunks)
                    judge_text = judge_compare(chat, doc_id=doc_id, query=query, weav_ctx=weav_ctx, pg_ctx=pg_ctx)
                    judge_text = remove_think_blocks(judge_text)

                    parsed: dict[str, object] = {}
                    json_text = extract_first_json_object(judge_text)
                    if json_text:
                        try:
                            parsed_json = json.loads(json_text)
                            if isinstance(parsed_json, dict):
                                parsed = parsed_json
                        except json.JSONDecodeError:
                            parsed = {}

                    postgres_score = clamp_percent(parsed.get("postgres_score"))
                    weaviate_score = clamp_percent(parsed.get("weaviate_score"))

                    item = {
                        "doc_id": doc_id,
                        "query": query,
                        "postgres_score": postgres_score,
                        "weaviate_score": weaviate_score,
                        "postgres_speed": pg_speed,
                        "weaviate_speed": weav_speed,
                    }
                    writer.writerow(item)
                    out_fh.flush()
                    print(json.dumps(item, ensure_ascii=False, indent=2))
                    log.info(
                        "Compared retrieval idx=%d doc_id=%s postgres_score=%d weaviate_score=%d postgres_speed=%.6f weaviate_speed=%.6f",
                        idx,
                        doc_id,
                        postgres_score,
                        weaviate_score,
                        pg_speed,
                        weav_speed,
                    )
                    if not json_text:
                        log.warning("Judge output has no JSON object for idx=%d: %s", idx, judge_text)
        print(
            json.dumps(
                {
                    "dataset_file": DATASET_FILE,
                    "results_file": RESULTS_FILE,
                    "appointments_total": len(dataset_rows),
                    "queries_total": queries_total,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        weav_store.close()
        pg_adapter.close()


if __name__ == "__main__":
    main()
