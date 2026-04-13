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


def main() -> int:
    try:
        from app.app import ingest_all_guidelines_from_env
        from app.config import get_settings
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Run 'pip install -r requirements.txt'.")
        return 1

    try:
        settings = get_settings()
        results = ingest_all_guidelines_from_env()
    except Exception as exc:
        print(f"Guideline ingestion failed: {exc}")
        return 1

    print(
        "Ingestion finished "
        f"| dir={settings.guidelines_dir} "
        f"| pattern={settings.guidelines_glob} "
        f"| clear_previous={settings.ingest_clear_previous} "
        f"| files={len(results)}"
    )
    for item in results:
        print(
            f"- document_id={item.document_id} "
            f"sections={item.sections_count} chunks={item.chunks_count} rules={item.rules_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
