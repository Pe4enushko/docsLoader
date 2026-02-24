from __future__ import annotations

from pathlib import Path

from engine.config import Settings  # noqa: F401  # loads .env
from engine.postgres import connect_postgres

INIT_SQL_PATH = Path("sql/init_medkard.sql")


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
