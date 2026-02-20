from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(slots=True)
class Settings:
    weaviate_url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    ollama_embed_base_url: str | None = os.getenv("OLLAMA_EMBED_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    ollama_chat_base_url: str | None = os.getenv("OLLAMA_CHAT_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    ingest_checkpoint_file: str = os.getenv("INGEST_CHECKPOINT_FILE", ".graphrag_ingest_checkpoint.json")
    k_initial: int = 32
    k_top: int = 12
    k_expand: int = 8
    packed_min: int = 6
    packed_max: int = 12
