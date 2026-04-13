#!/usr/bin/env python3
from __future__ import annotations

"""Run visit-audit pipeline on test data file and export results to JSON.

Environment variables:
- TEST_DATA_INPUT_PATH
- TEST_DATA_OUTPUT_PATH
- TEST_DATA_CONTINUE_ON_ERROR
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.integrations.one_c import extract_visit_guid, parse_appointments_payload


def _load_json(path: Path) -> Any:
    """Load JSON payload from file path."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_visits(payload: Any) -> list[dict[str, Any]]:
    """Normalize supported input payload formats to list of visit dictionaries."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        appointments = payload.get("appointments")
        if isinstance(appointments, list):
            return [item for item in appointments if isinstance(item, dict)]

        visits = payload.get("visits")
        if isinstance(visits, list):
            return [item for item in visits if isinstance(item, dict)]

        raw_payload = payload.get("raw_payload")
        if raw_payload is not None:
            return parse_appointments_payload(raw_payload)

        return parse_appointments_payload(payload)

    raise ValueError(f"Unsupported input JSON type: {type(payload).__name__}")


def main() -> int:
    settings = get_settings()
    input_path = Path(settings.test_data_input_path)
    output_path = Path(settings.test_data_output_path)

    try:
        payload = _load_json(input_path)
        visits = _as_visits(payload)
    except Exception as exc:
        print(f"Failed to read input data: {exc}")
        return 1

    if not visits:
        print(f"No visits found in input: {input_path}")
        return 1

    try:
        from app.models.db import SessionLocal
        from app.pipelines.visit_audit import VisitAuditPipeline
        from app.rag.postgres_adapter import PostgresRetrievalAdapter
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Run 'pip install -r requirements.txt'.")
        return 1

    started = time.time()
    items: list[dict[str, Any]] = []

    with SessionLocal() as session:
        retrieval = PostgresRetrievalAdapter(session)
        pipeline = VisitAuditPipeline(session=session, retrieval_adapter=retrieval)

        for idx, visit in enumerate(visits, start=1):
            external_id = extract_visit_guid(visit) or str(visit.get("id") or visit.get("guid") or "") or None
            try:
                result = pipeline.process_one(raw_visit=visit, external_id=external_id)
                session.commit()
                items.append(
                    {
                        "index": idx,
                        "external_id": external_id,
                        "status": "ok",
                        "visit_id": result.visit_id,
                        "report_id": result.report_id,
                        "pipeline_status": result.status,
                    }
                )
            except Exception as exc:
                session.rollback()
                items.append(
                    {
                        "index": idx,
                        "external_id": external_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if not settings.test_data_continue_on_error:
                    break

    ok_count = sum(1 for item in items if item.get("status") == "ok")
    err_count = sum(1 for item in items if item.get("status") == "error")

    output = {
        "meta": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total": len(items),
            "ok": ok_count,
            "errors": err_count,
            "duration_sec": round(time.time() - started, 3),
        },
        "results": items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "Test data audit finished "
        f"| total={len(items)} ok={ok_count} errors={err_count} "
        f"| output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
