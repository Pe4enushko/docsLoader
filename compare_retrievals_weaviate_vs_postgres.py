from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from langchain_ollama import ChatOllama

from engine.config import Settings
from engine.graphrag import PgvectorAgeAdapter, RetrievalService, WeaviateKnowledgeGraphAdapter, build_graphrag_postgres_dsn
from engine.logging_utils import setup_logging
from engine.models import ChunkRecord
from engine.weaviate import WeaviateGraphStore

DATASET_FILE = os.getenv("RETRIEVAL_DATASET_FILE", "retrieval_dataset.generated.json")
RESULTS_FILE = os.getenv("RETRIEVAL_COMPARE_RESULTS", "retrieval_compare_results.generated.json")
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


def judge_compare(chat: ChatOllama, doc_id: str, query: str, weav_ctx: list[dict], pg_ctx: list[dict]) -> str:
    prompt = (
        "Ты судья качества retrieval-контекста для медицинского RAG.\n"
        "Сравни два списка фрагментов и верни JSON:\n"
        '{ "winner": "weaviate|postgres|tie", "reason": "...", "answer": "краткий ответ на запрос пользователя по лучшему контексту" }\n\n'
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

        results = []
        idx = 0
        for row in dataset_rows:
            doc_id = row["doc_id"]
            visit_guid = row["visit_guid"]
            mkb_codes = row["mkb_codes"]

            for q_idx, query in enumerate(row["queries"], start=1):
                idx += 1
                weav_chunks = weav_retrieval.retrieve_context(doc_id=doc_id, query=query)
                pg_chunks = pg_retrieval.retrieve_context(doc_id=doc_id, query=query)

                weav_ctx = to_context(weav_chunks)
                pg_ctx = to_context(pg_chunks)
                judge_text = judge_compare(chat, doc_id=doc_id, query=query, weav_ctx=weav_ctx, pg_ctx=pg_ctx)

                item = {
                    "idx": idx,
                    "visit_guid": visit_guid,
                    "mkb_codes": mkb_codes,
                    "doc_id": doc_id,
                    "query_idx": q_idx,
                    "query": query,
                    "weaviate_chunks": len(weav_chunks),
                    "postgres_chunks": len(pg_chunks),
                    "judge": judge_text,
                }
                results.append(item)
                print(json.dumps(item, ensure_ascii=False, indent=2))
                log.info(
                    "Compared retrieval idx=%d visit_guid=%s doc_id=%s q_idx=%d weaviate=%d postgres=%d",
                    idx,
                    visit_guid,
                    doc_id,
                    q_idx,
                    len(weav_chunks),
                    len(pg_chunks),
                )

        Path(RESULTS_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "dataset_file": DATASET_FILE,
                    "results_file": RESULTS_FILE,
                    "appointments_total": len(dataset_rows),
                    "queries_total": len(results),
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
