from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ONE_C_APPOINTMENTS_URL = os.getenv("ONE_C_APPOINTMENTS_URL", "")
ONE_C_LOGIN = os.getenv("ONE_C_LOGIN", "")
ONE_C_PASSWORD = os.getenv("ONE_C_PASSWORD", "")
ONE_C_TIMEOUT_SECONDS = float(os.getenv("ONE_C_TIMEOUT_SECONDS", "15"))
OUTPUT_FILE = os.getenv("ONE_C_TEST_OUTPUT_FILE", "one_c_response.json")


def main() -> None:
    if not ONE_C_APPOINTMENTS_URL:
        raise ValueError("ONE_C_APPOINTMENTS_URL is not set")
    if not ONE_C_LOGIN or not ONE_C_PASSWORD:
        raise ValueError("ONE_C_LOGIN and ONE_C_PASSWORD must be set")

    token = base64.b64encode(f"{ONE_C_LOGIN}:{ONE_C_PASSWORD}".encode("utf-8")).decode("ascii")
    current_day = datetime.now().strftime("%d.%m.%Y")
    query = urllib.parse.urlencode({"datebegin": current_day, "dateend": current_day})
    separator = "&" if "?" in ONE_C_APPOINTMENTS_URL else "?"
    request_url = f"{ONE_C_APPOINTMENTS_URL}{separator}{query}"

    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json", "Authorization": f"Basic {token}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=ONE_C_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8")
        status = response.status

    try:
        payload = json.loads(body)
        out = json.dumps(payload, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        out = body

    Path(OUTPUT_FILE).write_text(out, encoding="utf-8")
    print(f"OK status={status} saved={OUTPUT_FILE}")


if __name__ == "__main__":
    main()

