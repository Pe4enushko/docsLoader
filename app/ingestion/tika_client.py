from __future__ import annotations

from pathlib import Path

import requests

from app.config import get_settings
from app.utils.logging import get_logger


log = get_logger(__name__)


class TikaClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: int | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.tika_url).rstrip("/")
        self.timeout = timeout_seconds or settings.tika_timeout_seconds

    def parse_to_text(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        url = f"{self.base_url}/tika"
        headers = {"Accept": "text/plain"}
        with file_path.open("rb") as fh:
            response = requests.put(url, data=fh, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        text = response.text
        log.info("Parsed document via Tika | path=%s | chars=%s", file_path, len(text))
        return text
