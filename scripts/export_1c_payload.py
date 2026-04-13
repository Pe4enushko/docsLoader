#!/usr/bin/env python3
from __future__ import annotations

"""Export 1C payload for a date range from env vars into local JSON file.

Expected env variables:
- ONE_C_DATE_BEGIN (dd.mm.yyyy)
- ONE_C_DATE_END (dd.mm.yyyy)
- ONE_C_EXPORT_PATH
- ONE_C_APPOINTMENTS_URL / ONE_C_LOGIN / ONE_C_PASSWORD
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from app.config import get_settings
from app.integrations.one_c import OneCClient, parse_appointments_payload
from app.utils.logging import configure_logging, get_logger


DATE_FMT = "%d.%m.%Y"
configure_logging()
log = get_logger(__name__)


def _resolve_dates() -> tuple[str, str]:
    settings = get_settings()
    date_begin = settings.one_c_date_begin.strip()
    date_end = settings.one_c_date_end.strip()

    if not date_begin:
        date_begin = datetime.now().strftime(DATE_FMT)
    if not date_end:
        date_end = date_begin

    # Validate format early to avoid confusing 1C-side errors.
    datetime.strptime(date_begin, DATE_FMT)
    datetime.strptime(date_end, DATE_FMT)
    return date_begin, date_end


def main() -> int:
    log.info("Script started | script=export_1c_payload")
    try:
        settings = get_settings()
        logging.getLogger().setLevel(getattr(logging, settings.log_level, logging.INFO))
        date_begin, date_end = _resolve_dates()
        log.info("Date range resolved | date_begin=%s | date_end=%s", date_begin, date_end)

        client = OneCClient.from_env()
        log.info("Fetching payload from 1C")
        payload, status = client.fetch_payload(date_begin=date_begin, date_end=date_end)
        log.info("1C payload fetched | http_status=%s", status)

        log.info("Normalizing appointments from raw payload")
        appointments = parse_appointments_payload(payload)
        log.info("Appointments normalized | count=%s", len(appointments))

        output_path = Path(settings.one_c_export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "meta": {
                "date_begin": date_begin,
                "date_end": date_end,
                "status": status,
                "count": len(appointments),
            },
            "raw_payload": payload,
            "appointments": appointments,
        }
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Output written | path=%s", output_path)
        print(f"Saved 1C export to {output_path} | appointments={len(appointments)} | status={status}")
        return 0
    except Exception as exc:
        log.exception("1C export script failed")
        print(f"1C export failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
