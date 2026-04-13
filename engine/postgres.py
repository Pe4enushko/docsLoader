from __future__ import annotations

import logging
import os
from urllib.parse import quote_plus

try:
    import psycopg  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None

try:
    import psycopg2  # type: ignore
except ImportError:  # pragma: no cover
    psycopg2 = None

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")
log = logging.getLogger(__name__)


def build_postgres_dsn() -> str:
    if not POSTGRES_HOST:
        raise ValueError("POSTGRES_HOST is not set")
    if not POSTGRES_PORT:
        raise ValueError("POSTGRES_PORT is not set")
    if not POSTGRES_USER:
        raise ValueError("POSTGRES_USER is not set")
    if not POSTGRES_PASSWORD:
        raise ValueError("POSTGRES_PASSWORD is not set")
    if not POSTGRES_DB:
        raise ValueError("POSTGRES_DB is not set")

    user = quote_plus(POSTGRES_USER)
    password = quote_plus(POSTGRES_PASSWORD)
    host = POSTGRES_HOST.strip()
    port = str(POSTGRES_PORT).strip()
    db = POSTGRES_DB.strip()
    sslmode = POSTGRES_SSLMODE.strip() or "prefer"
    return f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={sslmode}"


def connect_postgres():
    dsn = build_postgres_dsn()
    host = POSTGRES_HOST.strip()
    port = str(POSTGRES_PORT).strip()
    db = POSTGRES_DB.strip()
    sslmode = POSTGRES_SSLMODE.strip() or "prefer"
    try:
        if psycopg is not None:
            log.info("Connecting to Postgres via psycopg host=%s port=%s db=%s sslmode=%s", host, port, db, sslmode)
            return psycopg.connect(dsn)
        if psycopg2 is not None:
            log.info("Connecting to Postgres via psycopg2 host=%s port=%s db=%s sslmode=%s", host, port, db, sslmode)
            return psycopg2.connect(dsn)
        raise RuntimeError("Neither psycopg nor psycopg2 is installed")
    except Exception:
        log.exception("Postgres connection failed host=%s port=%s db=%s sslmode=%s", host, port, db, sslmode)
        raise


# ---------------------------------------------------------------------------
# Schema-specific helpers and query functions to be added for new schema.
# ---------------------------------------------------------------------------
