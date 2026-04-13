from __future__ import annotations

import re

from app.utils.text import normalize_space, strip_control_chars


class ClinicalTextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""
        payload = strip_control_chars(text)
        payload = payload.replace("\u00a0", " ")
        payload = re.sub(r"[ \t]+", " ", payload)
        payload = re.sub(r"\n{3,}", "\n\n", payload)
        return payload.strip()

    def clean_inline(self, text: str) -> str:
        return normalize_space(strip_control_chars(text))
