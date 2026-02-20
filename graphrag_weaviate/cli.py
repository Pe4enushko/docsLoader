from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from .config import Settings


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG for medical guidelines (Weaviate + Ollama)")
    parser.add_argument("--weaviate-url", default=None)
    parser.add_argument("--weaviate-api-key", default=None)
    parser.add_argument("--ollama-embed-model", default=None)
    parser.add_argument("--ollama-chat-model", default=None)
    parser.add_argument("--ollama-embed-base-url", default=None)
    parser.add_argument("--ollama-chat-base-url", default=None)
    parser.add_argument("--ollama-base-url", default=None, help="fallback for both embed/chat")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Batch ingest PDFs")
    ingest.add_argument("--input", required=True, dest="input_dir")
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--checkpoint", default=None)

    query = sub.add_parser("query", help="Retrieve packed chunks")
    query.add_argument("--doc_id", required=True)
    query.add_argument("--text", required=True)
    query.add_argument("--section_prefix", default=None)
    query.add_argument("--chunk_types", default=None, help="comma-separated allowlist")
    query.add_argument("--page_start", type=int, default=None)
    query.add_argument("--page_end", type=int, default=None)

    judge = sub.add_parser("judge", help="Evaluate doctor verdict")
    judge.add_argument("--doc_id", required=True)
    judge.add_argument("--verdict", required=True)

    return parser


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.weaviate_url:
        settings.weaviate_url = args.weaviate_url
    if args.weaviate_api_key:
        settings.weaviate_api_key = args.weaviate_api_key
    if args.ollama_embed_model:
        settings.ollama_embed_model = args.ollama_embed_model
    if args.ollama_chat_model:
        settings.ollama_chat_model = args.ollama_chat_model
    if args.ollama_embed_base_url:
        settings.ollama_embed_base_url = args.ollama_embed_base_url
    if args.ollama_chat_base_url:
        settings.ollama_chat_base_url = args.ollama_chat_base_url
    if args.ollama_base_url:
        if not settings.ollama_embed_base_url:
            settings.ollama_embed_base_url = args.ollama_base_url
        if not settings.ollama_chat_base_url:
            settings.ollama_chat_base_url = args.ollama_base_url
    return settings


def _chunk_to_output(c: Any) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "section_path": c.section_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type,
        "chunk_text": c.chunk_text,
        "score": c.score,
    }


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    settings = _build_settings(args)
    if args.cmd == "ingest" and args.checkpoint:
        settings.ingest_checkpoint_file = args.checkpoint

    t0 = time.time()
    from .ingestion import IngestionService
    from .judge import VerdictJudge
    from .retrieval import RetrievalService
    from .storage import WeaviateGraphStore

    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)

        if args.cmd == "ingest":
            ingester = IngestionService(store, checkpoint_file=settings.ingest_checkpoint_file)
            summary = ingester.ingest(input_dir=args.input_dir, manifest_path=args.manifest)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

        elif args.cmd == "query":
            page_range = None
            if args.page_start is not None and args.page_end is not None:
                page_range = (args.page_start, args.page_end)
            chunk_types = [x.strip() for x in args.chunk_types.split(",")] if args.chunk_types else None
            packed = retrieval.retrieve_context(
                doc_id=args.doc_id,
                query=args.text,
                section_prefix=args.section_prefix,
                chunk_type_allowlist=chunk_types,
                page_range=page_range,
            )
            print(
                json.dumps(
                    {
                        "doc_id": args.doc_id,
                        "query": args.text,
                        "count": len(packed),
                        "chunks": [_chunk_to_output(c) for c in packed],
                        "latency_sec": round(time.time() - t0, 3),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        elif args.cmd == "judge":
            judge = VerdictJudge(store, retrieval, settings)
            result = judge.evaluate_verdict(doc_id=args.doc_id, verdict_text=args.verdict)
            print(
                json.dumps(
                    {
                        "doc_id": args.doc_id,
                        "verdict": result.verdict.value,
                        "explanation": result.explanation,
                        "citations": result.citations,
                        "missing_info": result.missing_info,
                        "recommended_action": result.recommended_action,
                        "latency_sec": round(time.time() - t0, 3),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
