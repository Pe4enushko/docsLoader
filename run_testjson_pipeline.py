from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluateVerdict import (
    MANIFEST_PATH,
    SCORES_SYSTEM_PROMPT,
    build_row_for_medkard,
    extract_visit_guid,
    load_manifest_mkb_index,
)
from graphrag_weaviate.config import Settings
from graphrag_weaviate.llm import AppointmentJudge
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore

TEST_JSON_PATH = "testjson.json"
TEST_LOG_FILE = "logs/run_testjson_pipeline.log"


def load_appointments_from_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        appointments = payload.get("appointments")
        if isinstance(appointments, list):
            return [item for item in appointments if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("testjson must be object, appointments wrapper, or array")


def main() -> None:
    setup_logging(TEST_LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = AppointmentJudge(retrieval=retrieval, settings=settings, system_prompt=SCORES_SYSTEM_PROMPT)
        manifest_exact, manifest_group = load_manifest_mkb_index(MANIFEST_PATH)
        appointments = load_appointments_from_file(TEST_JSON_PATH)

        processed = 0
        for item in appointments:
            visit_guid = extract_visit_guid(item)
            if not visit_guid:
                raise ValueError("Missing Прием.GUID for appointment")

            build_row_for_medkard(
                judge=judge,
                appointment=item,
                manifest_exact=manifest_exact,
                manifest_group=manifest_group,
            )
            processed += 1

        print(json.dumps({"source": TEST_JSON_PATH, "processed": processed, "db_write": False}, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("testjson pipeline failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
