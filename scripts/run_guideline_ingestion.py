#!/usr/bin/env python3
from __future__ import annotations

"""Run batch guideline ingestion from env-configured folder.

Flow:
1. Optionally clear previous guideline ingestion data in DB.
2. Find all PDF files in GUIDELINES_DIR (pattern + recursion from env).
3. Run ingestion pipeline for each file.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from app.logger import configure_logging, get_logger


configure_logging()
log = get_logger(__name__)


def main() -> int:
    log.info("Script started | script=run_guideline_ingestion")
    try:
        from app.app import ingest_all_guidelines_from_env
        from app.config import get_settings
    except ModuleNotFoundError as exc:
        log.exception("Script failed during imports")
        print(f"Missing dependency: {exc}. Run 'pip install -r requirements.txt'.")
        return 1

    try:
        settings = get_settings()
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
        log.info(
            "Script parameters resolved | dir=%s | pattern=%s | recursive=%s | clear_previous=%s",
            settings.guidelines_dir,
            settings.guidelines_glob,
            settings.guidelines_recursive,
            settings.ingest_clear_previous,
        )
        results = ingest_all_guidelines_from_env()
    except Exception as exc:
        log.exception("Guideline ingestion script failed")
        print(f"Guideline ingestion failed: {exc}")
        return 1

    log.info("Script completed | ingested_files=%s", len(results))
    print(
        "Ingestion finished "
        f"| dir={settings.guidelines_dir} "
        f"| pattern={settings.guidelines_glob} "
        f"| clear_previous={settings.ingest_clear_previous} "
        f"| files={len(results)}"
    )
    for item in results:
        log.info(
            "Ingestion result item | doc_id=%s | sections=%s | chunks=%s | rules=%s",
            item.document_id,
            item.sections_count,
            item.chunks_count,
            item.rules_count,
        )
        print(
            f"- document_id={item.document_id} "
            f"sections={item.sections_count} chunks={item.chunks_count} rules={item.rules_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
