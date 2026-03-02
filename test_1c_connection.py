from __future__ import annotations

import json
import os
from pathlib import Path

from engine.integrations.one_c import OneCClient, parse_appointments_payload

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

OUTPUT_FILE = os.getenv("ONE_C_TEST_OUTPUT_FILE", "one_c_response.json")


def main() -> None:
    client = OneCClient.from_env()
    payload, status = client.fetch_payload_for_period(datebegin="26.02.2026", dateend="01.02.2024")
    appointments = parse_appointments_payload(payload)
    out = json.dumps(payload, ensure_ascii=False, indent=2)

    Path(OUTPUT_FILE).write_text(out, encoding="utf-8")
    print(f"OK status={status} appointments={len(appointments)} saved={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
