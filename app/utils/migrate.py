from __future__ import annotations

"""SQL migration runner for PostgreSQL.

This module replaces Alembic for this project. It applies plain SQL files from
`sql/migrations` in lexical order on every run.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.logger import configure_logging, get_logger


log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "sql" / "migrations"


@dataclass(slots=True, frozen=True)
class MigrationFile:
    """Represents one SQL migration file on disk."""

    version: str
    path: Path
    sql: str


class SQLMigrator:
    """Applies SQL migration scripts in lexical order."""

    def __init__(self, engine: Engine, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> None:
        self.engine = engine
        self.migrations_dir = migrations_dir

    def run(self) -> list[str]:
        """Apply all migration scripts and return executed file versions."""
        migrations = self._load_migrations()
        executed: list[str] = []
        for migration in migrations:
            self._apply_migration(migration)
            executed.append(migration.version)

        log.info("Migration run completed | executed=%s", len(executed))
        return executed

    def _load_migrations(self) -> list[MigrationFile]:
        """Read migration files from disk."""
        if not self.migrations_dir.exists():
            raise FileNotFoundError(f"Migration directory not found: {self.migrations_dir}")

        files = sorted(self.migrations_dir.glob("*.sql"))
        migrations: list[MigrationFile] = []
        for path in files:
            sql = path.read_text(encoding="utf-8")
            migrations.append(MigrationFile(version=path.name, path=path, sql=sql))
        return migrations

    def _apply_migration(self, migration: MigrationFile) -> None:
        """Apply one migration script in a transaction."""
        log.info("Applying migration | version=%s | path=%s", migration.version, migration.path)
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute(migration.sql)
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
            print("Executed migration scripts:")
            for version in applied:
                print(f"- {version}")
        else:
            print("No SQL scripts found.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
