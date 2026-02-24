from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_ollama import ChatOllama

from engine.config import Settings
from engine.graphrag import PgvectorAgeAdapter, RetrievalService, WeaviateKnowledgeGraphAdapter, build_graphrag_postgres_dsn
from engine.logging_utils import setup_logging
from engine.models import ChunkRecord
from engine.weaviate import WeaviateGraphStore

QUERIES_FILE = "retrieval_test_queries.txt"
LOG_FILE = "logs/compare_retrievals_weaviate_vs_postgres.log"


def load_queries(path: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            doc_id, query = line.split("\t", 1)
        elif "|" in line:
            doc_id, query = line.split("|", 1)
        else:
            raise ValueError(f"Expected 'doc_id<TAB>query' or 'doc_id|query' in {path}: {line}")
        out.append((doc_id.strip(), query.strip()))
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
    queries = load_queries(QUERIES_FILE)

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
        for idx, (doc_id, query) in enumerate(queries, start=1):
            weav_chunks = weav_retrieval.retrieve_context(doc_id=doc_id, query=query)
            pg_chunks = pg_retrieval.retrieve_context(doc_id=doc_id, query=query)

            weav_ctx = to_context(weav_chunks)
            pg_ctx = to_context(pg_chunks)
            judge_text = judge_compare(chat, doc_id=doc_id, query=query, weav_ctx=weav_ctx, pg_ctx=pg_ctx)

            item = {
                "idx": idx,
                "doc_id": doc_id,
                "query": query,
                "weaviate_chunks": len(weav_chunks),
                "postgres_chunks": len(pg_chunks),
                "judge": judge_text,
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False, indent=2))
            log.info("Compared retrieval idx=%d doc_id=%s weaviate=%d postgres=%d", idx, doc_id, len(weav_chunks), len(pg_chunks))

        print(json.dumps({"queries_total": len(results)}, ensure_ascii=False, indent=2))
    finally:
        weav_store.close()
        pg_adapter.close()


if __name__ == "__main__":
    main()
