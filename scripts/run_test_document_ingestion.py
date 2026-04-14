#!/usr/bin/env python3
from __future__ import annotations

"""Run detailed step-by-step test ingestion for one or many guideline documents.

Modes:
- single-source mode (`--source`): ingest exactly one explicit file,
- random-batch mode (default): pick N random files from the same folder/pattern
  used by main ingestion pipeline (`GUIDELINES_DIR`, `GUIDELINES_GLOB`,
  `GUIDELINES_RECURSIVE`).

For every document the script:
- logs start/end of each pipeline stage,
- captures exact failed stage with traceback,
- rolls back DB transaction on failure,
- includes full step telemetry in output JSON.
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ingestion import (
    ClinicalTextCleaner,
    GuidelineChunker,
    GuidelineRuleExtractor,
    GuidelineSectionNormalizer,
    TikaClient,
)
from app.models.db import SessionLocal
from app.repositories.guideline_repository import GuidelineRepository
from app.services.embedding import create_embedding_provider
from app.utils.logging import configure_logging, get_pipeline_logger


configure_logging()
log = get_pipeline_logger(__name__, "test_document_ingestion.log")


@dataclass(slots=True)
class StepRun:
    step: str
    status: str
    elapsed_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class TestIngestionRunResult:
    source_path: str
    status: str
    document_id: str | None
    total_elapsed_ms: float
    steps: list[StepRun] = field(default_factory=list)
    error: str | None = None


class TestDocumentIngestionRunner:
    """Execute one ingestion with explicit stage telemetry and failure tracking."""

    def __init__(self) -> None:
        self.tika_client = TikaClient()
        self.cleaner = ClinicalTextCleaner()
        self.normalizer = GuidelineSectionNormalizer()
        self.chunker = GuidelineChunker()
        self.rule_extractor = GuidelineRuleExtractor()
        self.embedding_provider = create_embedding_provider()
        log.info(
            "Test ingestion runner initialized | embedding_provider=%s | rule_extractor=%s",
            self.embedding_provider.__class__.__name__,
            self.rule_extractor.__class__.__name__,
        )

    def run(self, source_path: str) -> TestIngestionRunResult:
        """Run full ingestion for one source file with detailed step statuses."""
        source_file = _resolve_path(source_path)
        steps: list[StepRun] = []
        started = perf_counter()
        document_id: str | None = None

        if not source_file.exists() or not source_file.is_file():
            return TestIngestionRunResult(
                source_path=str(source_file),
                status="error",
                document_id=None,
                total_elapsed_ms=0.0,
                steps=[StepRun(step="validate_source", status="error", elapsed_ms=0.0, error="File not found")],
                error=f"File not found: {source_file}",
            )

        session = SessionLocal()
        repo = GuidelineRepository(session)

        log.info("Test ingestion started | file=%s", source_file)
        try:
            extracted_text = self._run_step(
                steps=steps,
                step_name="parse_tika",
                fn=lambda: self.tika_client.parse_to_text(source_file),
                details_fn=lambda text: {"chars": len(text)},
            )
            cleaned_text = self._run_step(
                steps=steps,
                step_name="clean_text",
                fn=lambda: self.cleaner.clean(extracted_text),
                details_fn=lambda text: {"chars": len(text)},
            )
            normalized_doc = self._run_step(
                steps=steps,
                step_name="normalize_sections",
                fn=lambda: self.normalizer.normalize(source_path=str(source_file), extracted_text=cleaned_text),
                details_fn=lambda doc: {"top_sections": len(doc.sections), "checksum": doc.checksum},
            )

            document = self._run_step(
                steps=steps,
                step_name="store_document_sections",
                fn=lambda: repo.upsert_document(normalized_doc),
                details_fn=lambda doc: {"document_id": str(doc.id), "sections_total": len(doc.sections)},
            )
            document_id = str(document.id)

            chunks = self._run_step(
                steps=steps,
                step_name="chunk_document",
                fn=lambda: self.chunker.chunk_document(normalized_doc),
                details_fn=lambda items: {"chunks": len(items)},
            )

            if chunks:
                embeddings = self._run_step(
                    steps=steps,
                    step_name="embed_chunks",
                    fn=lambda: self.embedding_provider.embed_texts([item.chunk_text for item in chunks]),
                    details_fn=lambda vectors: {
                        "chunks": len(vectors),
                        "vector_dim": len(vectors[0]) if vectors else 0,
                    },
                )
                for index, vector in enumerate(embeddings):
                    chunks[index].embedding = vector
            else:
                self._record_skipped(steps, "embed_chunks", {"reason": "No chunks produced"})

            stored_chunks = self._run_step(
                steps=steps,
                step_name="store_chunks",
                fn=lambda: repo.add_chunks(document, chunks),
                details_fn=lambda rows: {"stored_chunks": len(rows)},
            )

            rules = self._run_step(
                steps=steps,
                step_name="extract_rules",
                fn=lambda: self.rule_extractor.extract(normalized_doc),
                details_fn=lambda items: {"rules": len(items)},
            )
            stored_rules = self._run_step(
                steps=steps,
                step_name="store_rules",
                fn=lambda: repo.add_rules(document, rules),
                details_fn=lambda rows: {"stored_rules": len(rows)},
            )

            self._run_step(
                steps=steps,
                step_name="flush_session",
                fn=session.flush,
            )
            self._run_step(
                steps=steps,
                step_name="commit_transaction",
                fn=session.commit,
            )

            result = TestIngestionRunResult(
                source_path=str(source_file),
                status="ok",
                document_id=document_id,
                total_elapsed_ms=round((perf_counter() - started) * 1000, 1),
                steps=steps,
            )
            log.info(
                "Test ingestion completed | file=%s | document_id=%s | chunks=%s | rules=%s | elapsed_ms=%.1f",
                source_file,
                document_id,
                len(stored_chunks),
                len(stored_rules),
                result.total_elapsed_ms,
            )
            return result
        except Exception as exc:
            rollback_started = perf_counter()
            try:
                session.rollback()
                self._record_step(
                    steps,
                    StepRun(
                        step="rollback_transaction",
                        status="ok",
                        elapsed_ms=round((perf_counter() - rollback_started) * 1000, 1),
                        details={"reason": "Failure in previous step"},
                    ),
                )
            except Exception as rollback_exc:
                self._record_step(
                    steps,
                    StepRun(
                        step="rollback_transaction",
                        status="error",
                        elapsed_ms=round((perf_counter() - rollback_started) * 1000, 1),
                        error=str(rollback_exc),
                    ),
                )
                log.exception("Rollback failed after ingestion error | file=%s", source_file)

            result = TestIngestionRunResult(
                source_path=str(source_file),
                status="error",
                document_id=document_id,
                total_elapsed_ms=round((perf_counter() - started) * 1000, 1),
                steps=steps,
                error=str(exc),
            )
            log.exception("Test ingestion failed | file=%s | error=%s", source_file, exc)
            return result
        finally:
            session.close()

    def _run_step(
        self,
        *,
        steps: list[StepRun],
        step_name: str,
        fn: Callable[[], Any],
        details_fn: Callable[[Any], dict[str, Any]] | None = None,
    ) -> Any:
        """Execute one step with timing, structured logs and failure capture."""
        started = perf_counter()
        log.info("Test ingestion step started | step=%s", step_name)
        try:
            payload = fn()
        except Exception as exc:
            elapsed_ms = round((perf_counter() - started) * 1000, 1)
            step = StepRun(step=step_name, status="error", elapsed_ms=elapsed_ms, error=str(exc))
            self._record_step(steps, step)
            log.exception("Test ingestion step failed | step=%s | elapsed_ms=%.1f", step_name, elapsed_ms)
            raise

        elapsed_ms = round((perf_counter() - started) * 1000, 1)
        details = details_fn(payload) if details_fn else {}
        step = StepRun(step=step_name, status="ok", elapsed_ms=elapsed_ms, details=details)
        self._record_step(steps, step)
        log.info(
            "Test ingestion step done | step=%s | elapsed_ms=%.1f | details=%s",
            step_name,
            elapsed_ms,
            details,
        )
        return payload

    def _record_skipped(self, steps: list[StepRun], step_name: str, details: dict[str, Any]) -> None:
        """Record skipped step (e.g. embedding when no chunks exist)."""
        step = StepRun(step=step_name, status="skipped", elapsed_ms=0.0, details=details)
        self._record_step(steps, step)
        log.info("Test ingestion step skipped | step=%s | details=%s", step_name, details)

    def _record_step(self, steps: list[StepRun], step: StepRun) -> None:
        steps.append(step)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _discover_random_sources(*, docs_count: int, random_seed: int | None) -> tuple[list[Path], dict[str, Any]]:
    settings = get_settings()
    base_dir = _resolve_path(settings.guidelines_dir)
    pattern = settings.guidelines_glob
    recursive = settings.guidelines_recursive

    if not base_dir.exists():
        raise FileNotFoundError(f"GUIDELINES_DIR not found: {base_dir}")

    candidates = sorted(base_dir.rglob(pattern) if recursive else base_dir.glob(pattern))
    candidates = [item for item in candidates if item.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found by pattern '{pattern}' in {base_dir}")

    requested = max(1, int(docs_count))
    if requested > len(candidates):
        log.warning(
            "Requested docs count is larger than available files | requested=%s | available=%s",
            requested,
            len(candidates),
        )
    selected_count = min(requested, len(candidates))

    randomizer = random.Random(random_seed)
    selected = randomizer.sample(candidates, k=selected_count)

    meta = {
        "guidelines_dir": str(base_dir),
        "guidelines_glob": pattern,
        "guidelines_recursive": recursive,
        "available_files": len(candidates),
        "requested_docs_count": requested,
        "selected_docs_count": selected_count,
        "random_seed": random_seed,
    }
    return selected, meta


def _write_batch_report(payload: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    path = _resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Test ingestion batch report saved | path=%s", path)


def _build_arg_parser(settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test document ingestion with detailed step logs")
    parser.add_argument(
        "--source",
        default=settings.test_ingestion_source_path,
        help="Path to one source PDF/DOC/DOCX file (optional; if empty random batch mode is used)",
    )
    parser.add_argument(
        "--docs-count",
        type=int,
        default=settings.test_ingestion_docs_count,
        help="Number of random documents to test in batch mode (TEST_INGESTION_DOCS_COUNT)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=settings.test_ingestion_random_seed,
        help="Optional random seed for reproducible batch selection (TEST_INGESTION_RANDOM_SEED)",
    )
    parser.add_argument(
        "--output",
        default=settings.test_ingestion_output_path,
        help="Path to JSON batch report (or TEST_INGESTION_OUTPUT_PATH)",
    )
    return parser


def main() -> int:
    settings = get_settings()
    logging.getLogger().setLevel(getattr(logging, settings.log_level, logging.INFO))
    parser = _build_arg_parser(settings)
    args = parser.parse_args()

    source = str(args.source or "").strip()
    output = str(args.output or "").strip()

    runner = TestDocumentIngestionRunner()
    started = perf_counter()

    if source:
        selected_sources = [_resolve_path(source)]
        selection_meta = {
            "mode": "single_source",
            "requested_docs_count": 1,
            "selected_docs_count": 1,
            "random_seed": None,
        }
    else:
        try:
            selected_sources, discovery_meta = _discover_random_sources(
                docs_count=args.docs_count,
                random_seed=args.random_seed,
            )
        except Exception as exc:
            log.exception("Failed to select random sources for test ingestion")
            print(f"Failed to resolve random source files: {exc}")
            return 1
        selection_meta = {"mode": "random_batch", **discovery_meta}

    results: list[TestIngestionRunResult] = []
    for index, source_path in enumerate(selected_sources, start=1):
        log.info(
            "Test ingestion batch item started | index=%s/%s | source=%s",
            index,
            len(selected_sources),
            source_path,
        )
        result = runner.run(source_path=str(source_path))
        results.append(result)
        log.info(
            "Test ingestion batch item finished | index=%s/%s | source=%s | status=%s | document_id=%s",
            index,
            len(selected_sources),
            source_path,
            result.status,
            result.document_id,
        )

    ok_count = sum(1 for item in results if item.status == "ok")
    error_count = len(results) - ok_count
    report_payload = {
        "meta": {
            **selection_meta,
            "total": len(results),
            "ok": ok_count,
            "errors": error_count,
            "duration_sec": round(perf_counter() - started, 3),
            "output_path": str(_resolve_path(output)) if output else None,
        },
        "selected_sources": [str(path) for path in selected_sources],
        "results": [asdict(item) for item in results],
    }
    _write_batch_report(report_payload, output or None)

    print(
        "Test document ingestion finished "
        f"| mode={selection_meta.get('mode')} "
        f"| total={len(results)} ok={ok_count} errors={error_count} "
        f"| output={output or '(disabled)'}"
    )

    if error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
