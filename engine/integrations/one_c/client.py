from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any


def parse_appointments_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("HTTP response must be JSON object with 'appointments'")
    appointments = payload.get("appointments")
    if not isinstance(appointments, list):
        raise ValueError("HTTP response must contain 'appointments' array")
    return [item for item in appointments if isinstance(item, dict)]


class OneCClient:
    def __init__(
        self,
        url: str,
        login: str,
        password: str,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.url = url
        self.login = login
        self.password = password
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OneCClient":
        return cls(
            url=os.getenv("ONE_C_APPOINTMENTS_URL", ""),
            login=os.getenv("ONE_C_LOGIN", ""),
            password=os.getenv("ONE_C_PASSWORD", ""),
            timeout_seconds=float(os.getenv("ONE_C_TIMEOUT_SECONDS", "15")),
        )

    def fetch_payload_for_today(self) -> tuple[dict[str, Any], int]:
        if not self.url or self.url.startswith("<"):
            raise ValueError("Set real ONE_C_APPOINTMENTS_URL in environment")
        if not self.login or not self.password:
            raise ValueError("ONE_C_LOGIN and ONE_C_PASSWORD must be set")

        token = base64.b64encode(f"{self.login}:{self.password}".encode("utf-8")).decode("ascii")
        current_day = datetime.now().strftime("%d.%m.%Y")
        query_params = urllib.parse.urlencode({"datebegin": current_day, "dateend": current_day})
        separator = "&" if "?" in self.url else "?"
        request_url = f"{self.url}{separator}{query_params}"

        request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "Authorization": f"Basic {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload, int(response.status)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch appointments from 1C: {exc}") from exc

    def fetch_appointments_for_today(self) -> list[dict[str, Any]]:
        payload, _ = self.fetch_payload_for_today()
        return parse_appointments_payload(payload)

