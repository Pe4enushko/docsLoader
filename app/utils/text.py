from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")


def fix_mojibake(text: str) -> str:
    """Attempt to repair common UTF-8->Latin mojibake sequences.

    Typical symptom in Cyrillic texts:
    `Epidemiologiya` appears as `D0...`-style broken symbols.
    """
    if not text:
        return text

    original = text
    candidates = [original]

    # Most frequent corruption path: UTF-8 bytes interpreted as latin-1/cp1252.
    for wrong_encoding in ("latin1", "cp1252"):
        try:
            candidate = original.encode(wrong_encoding).decode("utf-8")
        except UnicodeError:
            continue
        if candidate:
            candidates.append(candidate)

    return max(candidates, key=_text_quality_score)


def _text_quality_score(text: str) -> int:
    """Heuristic score: prefer more Cyrillic and fewer mojibake markers."""
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    mojibake_markers = sum(text.count(marker) for marker in ("Ð", "Ñ", "Ã", "Â", "\ufffd"))
    return cyrillic - (mojibake_markers * 3)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
