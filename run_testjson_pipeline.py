from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluateVerdict import (
    MANIFEST_PATH,
    build_row_for_medkard,
    extract_visit_guid,
    load_manifest_mkb_index,
)
from engine.appointments import parse_appointments_payload
from engine.config import Settings
from engine.llm import AppointmentJudge
from engine.logging_utils import setup_logging
from engine.retrieval import RetrievalService
from engine.storage import WeaviateGraphStore

TEST_JSON_PATH = "testjson.json"
TEST_LOG_FILE = "logs/run_testjson_pipeline.log"


def load_appointments_from_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_appointments_payload(payload)


def main() -> None:
    setup_logging(TEST_LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = AppointmentJudge(retrieval=retrieval, settings=settings)
        manifest_exact, manifest_group, manifest_titles = load_manifest_mkb_index(MANIFEST_PATH)
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
                manifest_titles=manifest_titles,
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
