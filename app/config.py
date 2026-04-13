from __future__ import annotations

"""Centralized runtime settings for the medical audit platform.

All environment values are collected in one typed dataclass so service/pipeline
modules can depend on explicit configuration instead of ad-hoc `os.getenv` calls.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass(slots=True, frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "medical-audit-pipeline")
    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/med_audit",
    )
    db_echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"

    tika_url: str = os.getenv("TIKA_URL", "http://localhost:9998")
    tika_timeout_seconds: int = int(os.getenv("TIKA_TIMEOUT_SECONDS", "30"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_llm_base_url: str | None = os.getenv("OPENAI_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
    openai_embedding_base_url: str | None = os.getenv("OPENAI_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL"))

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
    ollama_embed_base_url: str = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    ollama_llm_base_url: str = os.getenv("OLLAMA_CHAT_BASE_URL", "http://localhost:11434")
    ollama_llm_model: str = os.getenv("OLLAMA_CHAT_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b"))
    ollama_llm_num_ctx: int = int(os.getenv("OLLAMA_CHAT_NUM_CTX", "16384"))

    # Optional override for rule extraction stage; falls back to general LLM model.
    rule_extractor_model: str = os.getenv("RULE_EXTRACTOR_MODEL", os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini"))
    rule_extractor_force_json_mode: bool = os.getenv("RULE_EXTRACTOR_FORCE_JSON_MODE", "true").lower() == "true"
    rule_extractor_save_raw_responses: bool = os.getenv("RULE_EXTRACTOR_SAVE_RAW_RESPONSES", "true").lower() == "true"
    rule_extractor_raw_dir: str = os.getenv("RULE_EXTRACTOR_RAW_DIR", "tmp/rule_extractor_raw")

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))

    # Batch ingestion settings.
    guidelines_dir: str = os.getenv("GUIDELINES_DIR", "docs")
    guidelines_glob: str = os.getenv("GUIDELINES_GLOB", "*.pdf")
    guidelines_recursive: bool = os.getenv("GUIDELINES_RECURSIVE", "true").lower() == "true"
    ingest_clear_previous: bool = os.getenv("INGEST_CLEAR_PREVIOUS", "true").lower() == "true"

    # 1C export script settings.
    one_c_date_begin: str = os.getenv("ONE_C_DATE_BEGIN", "")
    one_c_date_end: str = os.getenv("ONE_C_DATE_END", "")
    one_c_export_path: str = os.getenv("ONE_C_EXPORT_PATH", "tmp/one_c_export.json")

    # Test data audit script settings.
    test_data_input_path: str = os.getenv("TEST_DATA_INPUT_PATH", "tmp/test_visits_input.json")
    test_data_output_path: str = os.getenv("TEST_DATA_OUTPUT_PATH", "tmp/test_visits_output.json")
    test_data_continue_on_error: bool = os.getenv("TEST_DATA_CONTINUE_ON_ERROR", "true").lower() == "true"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings object for the current process."""
    return Settings()
