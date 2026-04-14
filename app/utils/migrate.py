from __future__ import annotations

"""SQL migration runner for PostgreSQL.

This module replaces Alembic for this project. It applies plain SQL files from
`sql/migrations` in lexical order and stores migration history in
`schema_migrations`.
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.utils.logging import configure_logging, get_logger


log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "sql" / "migrations"


@dataclass(slots=True, frozen=True)
class MigrationFile:
    """Represents one SQL migration file on disk."""

    version: str
    path: Path
    checksum: str
    sql: str


class SQLMigrator:
    """Applies SQL migrations and tracks applied versions in DB."""

    def __init__(self, engine: Engine, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> None:
        self.engine = engine
        self.migrations_dir = migrations_dir

    def run(self) -> list[str]:
        """Apply all pending migrations and return list of applied versions."""
        self._ensure_migrations_table()
        migrations = self._load_migrations()
        applied = self._load_applied_map()

        applied_now: list[str] = []
        for migration in migrations:
            prev_checksum = applied.get(migration.version)
            if prev_checksum:
                if prev_checksum != migration.checksum:
                    raise RuntimeError(
                        "Migration checksum mismatch for "
                        f"{migration.version}: db={prev_checksum} file={migration.checksum}"
                    )
                log.info("Migration already applied | version=%s", migration.version)
                continue

            self._apply_migration(migration)
            applied_now.append(migration.version)

        log.info("Migration run completed | applied=%s", len(applied_now))
        return applied_now

    def _ensure_migrations_table(self) -> None:
        """Create migration history table if it does not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version text PRIMARY KEY,
            checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))

    def _load_applied_map(self) -> dict[str, str]:
        """Return {version: checksum} map of applied migrations."""
        query = text("SELECT version, checksum FROM schema_migrations")
        with self.engine.connect() as conn:
            rows = conn.execute(query).all()
        return {str(row[0]): str(row[1]) for row in rows}

    def _load_migrations(self) -> list[MigrationFile]:
        """Read migration files from disk and calculate checksums."""
        if not self.migrations_dir.exists():
            raise FileNotFoundError(f"Migration directory not found: {self.migrations_dir}")

        files = sorted(self.migrations_dir.glob("*.sql"))
        migrations: list[MigrationFile] = []
        for path in files:
            sql = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            migrations.append(MigrationFile(version=path.name, path=path, checksum=checksum, sql=sql))
        return migrations

    def _apply_migration(self, migration: MigrationFile) -> None:
        """Apply one migration in a transaction and record it."""
        log.info("Applying migration | version=%s | path=%s", migration.version, migration.path)
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute(migration.sql)
                cur.execute(
                    "INSERT INTO schema_migrations(version, checksum) VALUES (%s, %s)",
                    (migration.version, migration.checksum),
                )
            raw_conn.commit()
            log.info("Migration applied | version=%s", migration.version)
        except Exception:
            raw_conn.rollback()
            log.exception("Migration failed | version=%s", migration.version)
            raise
        finally:
            raw_conn.close()


def run_migrations(migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> list[str]:
    """Convenience function used by app entry points."""
    settings = get_settings()
    engine = create_engine(settings.database_url, echo=settings.db_echo, future=True)
    migrator = SQLMigrator(engine=engine, migrations_dir=migrations_dir)
    try:
        return migrator.run()
    finally:
        engine.dispose()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQL migrations")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=DEFAULT_MIGRATIONS_DIR,
        help="Path to directory with *.sql migrations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    try:
        applied = run_migrations(migrations_dir=args.migrations_dir)
        if applied:
            print("Applied migrations:")
            for version in applied:
                print(f"- {version}")
        else:
            print("No pending migrations.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
