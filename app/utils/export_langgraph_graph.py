from __future__ import annotations

"""Utility for exporting VisitAuditPipeline LangGraph schema to PNG.

This tool is intentionally project-aware:
- uses the same settings/logging stack as runtime scripts,
- resolves default output path under configured log directory,
- builds graph from the actual pipeline wiring used in production.
"""

import argparse
from pathlib import Path

from app.config import get_settings
from app.models.db import SessionLocal
from app.pipelines.visit_audit import VisitAuditPipeline
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.utils.logging import configure_logging, get_pipeline_logger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for graph export."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export LangGraph schema for visit audit pipeline")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.log_dir) / "visit_audit_graph_schema.png",
        help="Target PNG path for rendered graph schema",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_pipeline_logger(__name__, "langgraph_export.log")

    args = _parse_args(argv)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("LangGraph export started | output=%s", output_path)

    with SessionLocal() as session:
        retrieval = PostgresRetrievalAdapter(session)
        pipeline = VisitAuditPipeline(session=session, retrieval_adapter=retrieval)
        graph_app = pipeline.get_graph()

        try:
            graph_app.get_graph().draw_png(output_file_path=str(output_path))
            log.info("LangGraph export completed | output=%s", output_path)
            print(f"Граф успешно сохранен в {output_path}")
            return 0
        except Exception as e:
            log.exception("LangGraph export failed | output=%s", output_path)
            print(f"Ошибка: {e}. Проверьте установку Graphviz в системе.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
