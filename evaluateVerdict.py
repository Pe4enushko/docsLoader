from __future__ import annotations

import json
import logging

from graphrag_weaviate.config import Settings
from graphrag_weaviate.judge import VerdictJudge
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore

# Main program: judge only.
DOC_ID_OR_MKB_CODE = "I50"
VERDICT_TEXT = "Назначить иАПФ и контроль креатинина через 2 недели"
CHECKPOINT_FILE = ".graphrag_ingest_checkpoint.json"
LOG_FILE = "logs/docsToGraphRAG.log"


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    settings.ingest_checkpoint_file = CHECKPOINT_FILE

    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = VerdictJudge(store, retrieval, settings)
        result = judge.evaluate_verdict(doc_id=DOC_ID_OR_MKB_CODE, verdict_text=VERDICT_TEXT)
        print(
            json.dumps(
                {
                    "doc_id": DOC_ID_OR_MKB_CODE,
                    "verdict": result.verdict.value,
                    "explanation": result.explanation,
                    "citations": result.citations,
                    "missing_info": result.missing_info,
                    "recommended_action": result.recommended_action,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        log.exception("docsToGraphRAG pipeline failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
