"""Project dependency list and helper exporter.

Usage:
  python3 requirements.py
This writes requirements.txt in the project root.
"""

from __future__ import annotations

from pathlib import Path

REQUIREMENTS = [
    "weaviate-client>=4.10.0",
    "langchain-ollama>=0.2.0",
    "pymupdf>=1.24.0",
    "python-dotenv>=1.0.1",
]


def export_requirements_txt(path: str = "requirements.txt") -> Path:
    out = Path(path)
    out.write_text("\n".join(REQUIREMENTS) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = export_requirements_txt()
    print(f"Wrote {p}")
