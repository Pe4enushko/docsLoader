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
    ollama_embed_base_url: str | None = os.getenv("OLLAMA_EMBED_BASE_URL")
    ollama_chat_base_url: str | None = os.getenv("OLLAMA_CHAT_BASE_URL")
    ollama_chat_num_ctx: int = int(os.getenv("OLLAMA_CHAT_NUM_CTX", "16384"))
    ingest_checkpoint_file: str = os.getenv("INGEST_CHECKPOINT_FILE", ".graphrag_ingest_checkpoint.json")
    k_initial: int = 24
    k_top: int = 8
    k_expand: int = 6
    packed_min: int = 4
    packed_max: int = 8
