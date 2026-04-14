from __future__ import annotations

from pathlib import Path

import requests

from app.config import get_settings
from app.logger import get_logger
from app.utils.text import fix_mojibake


log = get_logger(__name__)


class TikaClient:
    """Thin Apache Tika client with defensive decoding.

    Tika usually returns UTF-8 text, but in practice upstream/proxy layers may
    surface mojibake. We decode bytes explicitly and then run text repair.
    """

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

        raw_bytes = response.content
        try:
            decoded_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to requests decoding when upstream response is not utf-8.
            decoded_text = response.text

        text = fix_mojibake(decoded_text)
        if text != decoded_text:
            log.warning("Detected and repaired mojibake in Tika payload | path=%s", file_path)
        log.info("Parsed document via Tika | path=%s | chars=%s", file_path, len(text))
        return text
