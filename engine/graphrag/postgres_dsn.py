from __future__ import annotations

import os
from urllib.parse import quote_plus


def build_graphrag_postgres_dsn() -> str:
    host = os.getenv("GRAPHRAG_PG_HOST", "").strip()
    port = os.getenv("GRAPHRAG_PG_PORT", "5432").strip()
    user = os.getenv("GRAPHRAG_PG_USER", "").strip()
    password = os.getenv("GRAPHRAG_PG_PASSWORD", "")
    database = os.getenv("GRAPHRAG_PG_DB", "").strip()
    sslmode = os.getenv("GRAPHRAG_PG_SSLMODE", "prefer").strip() or "prefer"

    if not host:
        raise ValueError("GRAPHRAG_PG_HOST is not set")
    if not port:
        raise ValueError("GRAPHRAG_PG_PORT is not set")
    if not user:
        raise ValueError("GRAPHRAG_PG_USER is not set")
    if password == "":
        raise ValueError("GRAPHRAG_PG_PASSWORD is not set")
    if not database:
        raise ValueError("GRAPHRAG_PG_DB is not set")

    user_q = quote_plus(user)
    pass_q = quote_plus(password)
    return f"postgresql://{user_q}:{pass_q}@{host}:{port}/{database}?sslmode={sslmode}"


def masked_graphrag_dsn_for_logs(dsn: str) -> str:
    if "://" not in dsn or "@" not in dsn:
        return "<invalid_dsn>"
    scheme, rest = dsn.split("://", 1)
    creds, tail = rest.split("@", 1)
    if ":" not in creds:
        return f"{scheme}://***@{tail}"
    user, _ = creds.split(":", 1)
    return f"{scheme}://{user}:***@{tail}"
