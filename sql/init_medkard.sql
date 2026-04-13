-- Bootstrap SQL for manual environments.
-- Primary schema management is handled by Alembic migrations.

CREATE EXTENSION IF NOT EXISTS vector;

-- Recommended:
--   alembic upgrade head
-- This applies full DDL for guideline/normative/visit/audit tables.
