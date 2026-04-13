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
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    ollama_chat_base_url: str | None = os.getenv("OLLAMA_CHAT_BASE_URL")
    ollama_chat_num_ctx: int = int(os.getenv("OLLAMA_CHAT_NUM_CTX", "16384"))
