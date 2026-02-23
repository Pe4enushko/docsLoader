from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path

from graphrag_weaviate.config import Settings  # noqa: F401  # loads .env

try:
    import psycopg  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None

try:
    import psycopg2  # type: ignore
except ImportError:  # pragma: no cover
    psycopg2 = None

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
INIT_SQL_PATH = Path("sql/init_medkard.sql")


def extract_db_name_from_dsn(dsn: str) -> str | None:
    parsed = urllib.parse.urlparse(dsn)
    if parsed.scheme and parsed.path and parsed.path != "/":
        return parsed.path.lstrip("/")
    match = re.search(r"(?:^|\\s)(?:dbname|database)\\s*=\\s*([^\\s]+)", dsn)
    if match:
        return match.group(1).strip().strip("'\"")
    return None


def connect_postgres():
    if not POSTGRES_DSN:
        raise ValueError("POSTGRES_DSN is not set in .env")
    if not POSTGRES_DB:
        raise ValueError("POSTGRES_DB is not set in .env")

    db_from_dsn = extract_db_name_from_dsn(POSTGRES_DSN)
    if db_from_dsn and db_from_dsn != POSTGRES_DB:
        raise ValueError(f"POSTGRES_DB ({POSTGRES_DB}) does not match DB in POSTGRES_DSN ({db_from_dsn})")

    if psycopg is not None:
        return psycopg.connect(POSTGRES_DSN)
    if psycopg2 is not None:
        return psycopg2.connect(POSTGRES_DSN)
    raise RuntimeError("Neither psycopg nor psycopg2 is installed")


def main() -> None:
    if not INIT_SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {INIT_SQL_PATH}")
    sql = INIT_SQL_PATH.read_text(encoding="utf-8")
    with connect_postgres() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("MedKard table initialized")


if __name__ == "__main__":
    main()
