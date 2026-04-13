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

from app.config import get_settings
from app.integrations.one_c import OneCClient, parse_appointments_payload


DATE_FMT = "%d.%m.%Y"


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
    try:
        settings = get_settings()
        date_begin, date_end = _resolve_dates()

        client = OneCClient.from_env()
        payload, status = client.fetch_payload(date_begin=date_begin, date_end=date_end)
        appointments = parse_appointments_payload(payload)

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
        print(f"Saved 1C export to {output_path} | appointments={len(appointments)} | status={status}")
        return 0
    except Exception as exc:
        print(f"1C export failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
