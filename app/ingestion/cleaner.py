from __future__ import annotations

import re

from app.utils.text import fix_mojibake, normalize_space, strip_control_chars


class ClinicalTextCleaner:
    """Normalize extracted document text before section parsing.

    Responsibilities:
    - remove control/noise symbols,
    - repair common mojibake artifacts,
    - normalize whitespace for deterministic downstream parsing.
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""
        payload = fix_mojibake(text)
        payload = strip_control_chars(payload)
        payload = payload.replace("\u00a0", " ")
        payload = re.sub(r"[ \t]+", " ", payload)
        payload = re.sub(r"\n{3,}", "\n\n", payload)
        return payload.strip()

    def clean_inline(self, text: str) -> str:
        return normalize_space(strip_control_chars(fix_mojibake(text)))
