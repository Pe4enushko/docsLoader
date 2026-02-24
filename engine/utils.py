from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def estimate_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return int(words * 1.3)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated {len(text) - max_len} chars>"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
