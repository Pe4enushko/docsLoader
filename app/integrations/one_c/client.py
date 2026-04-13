from __future__ import annotations

"""HTTP client for retrieving appointment payloads from 1C endpoint."""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from app.integrations.one_c.parser import parse_appointments_payload


class OneCClient:
    """Minimal 1C client with basic-auth and date-range filtering."""

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
        """Initialize client from environment variables."""
        return cls(
            url=os.getenv("ONE_C_APPOINTMENTS_URL", ""),
            login=os.getenv("ONE_C_LOGIN", ""),
            password=os.getenv("ONE_C_PASSWORD", ""),
            timeout_seconds=float(os.getenv("ONE_C_TIMEOUT_SECONDS", "15")),
        )

    def fetch_payload(self, date_begin: str, date_end: str) -> tuple[Any, int]:
        """Fetch raw payload from 1C for date range `dd.mm.yyyy` -> `dd.mm.yyyy`."""
        if not self.url or self.url.startswith("<"):
            raise ValueError("Set real ONE_C_APPOINTMENTS_URL in environment")
        if not self.login or not self.password:
            raise ValueError("ONE_C_LOGIN and ONE_C_PASSWORD must be set")

        token = base64.b64encode(f"{self.login}:{self.password}".encode("utf-8")).decode("ascii")
        query_params = urllib.parse.urlencode({"datebegin": date_begin, "dateend": date_end})
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

    def fetch_payload_for_today(self) -> tuple[Any, int]:
        """Convenience method for requesting today's appointments only."""
        current_day = datetime.now().strftime("%d.%m.%Y")
        return self.fetch_payload(date_begin=current_day, date_end=current_day)

    def fetch_appointments(self, date_begin: str, date_end: str) -> list[dict[str, Any]]:
        """Fetch and normalize appointments for a date range."""
        payload, _ = self.fetch_payload(date_begin=date_begin, date_end=date_end)
        return parse_appointments_payload(payload)

    def fetch_appointments_for_today(self) -> list[dict[str, Any]]:
        """Fetch and normalize appointments for current day."""
        payload, _ = self.fetch_payload_for_today()
        return parse_appointments_payload(payload)
