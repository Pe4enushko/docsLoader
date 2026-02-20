from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphrag_weaviate.config import Settings

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:  # type: ignore[redef]
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize MKB knowledge graph by ingesting all clinical recommendation PDFs from a custom path"
    )
    parser.add_argument("--input", required=True, help="Folder with PDF files")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON to enrich metadata")
    parser.add_argument("--doc-prefix", default="mkb", help="Prefix for auto-generated doc_id/mkb_code")
    parser.add_argument("--checkpoint", default=None, help="Override ingest checkpoint file")
    parser.add_argument("--strict", action="store_true", help="Fail if a PDF is missing in manifest")
    return parser


def _slug(s: str) -> str:
    value = re.sub(r"[^\w\-]+", "_", s.strip(), flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower() or "document"


def _load_manifest(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        out: dict[str, dict[str, Any]] = {}
        for row in data:
            filename = row.get("filename") or row.get("file")
            if filename:
                out[str(filename)] = row
        return out
    if isinstance(data, dict):
        if "documents" in data and isinstance(data["documents"], list):
            out: dict[str, dict[str, Any]] = {}
            for row in data["documents"]:
                filename = row.get("filename") or row.get("file")
                if filename:
                    out[str(filename)] = row
            return out
        return {str(k): v for k, v in data.items()}
    raise ValueError("Unsupported manifest format")


def _build_meta(pdf_path: Path, prefix: str, manifest_map: dict[str, dict[str, Any]], strict: bool) -> dict[str, Any]:
    filename = pdf_path.name
    manifest_row = manifest_map.get(filename)

    if strict and manifest_map and not manifest_row:
        raise ValueError(f"Missing manifest entry for: {filename}")

    stem = _slug(pdf_path.stem)
    default_doc_id = f"{prefix}_{stem}"

    if manifest_row:
        return {
            "doc_id": str(manifest_row.get("doc_id") or default_doc_id),
            "title": str(manifest_row.get("title") or pdf_path.stem),
            "year": manifest_row.get("year"),
            "specialty": manifest_row.get("specialty"),
            "source_url": manifest_row.get("source_url"),
        }

    return {
        "doc_id": default_doc_id,
        "title": pdf_path.stem,
        "year": None,
        "specialty": None,
        "source_url": None,
    }


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()

    from graphrag_weaviate.ingestion import IngestionService
    from graphrag_weaviate.storage import WeaviateGraphStore

    settings = Settings()
    if args.checkpoint:
        settings.ingest_checkpoint_file = args.checkpoint

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input path does not exist or is not a directory: {input_dir}")

    manifest_map = _load_manifest(args.manifest)
    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files = sorted(input_dir.glob(pattern))

    store = WeaviateGraphStore(settings)
    try:
        ingester = IngestionService(store, checkpoint_file=settings.ingest_checkpoint_file)
        results = []

        for pdf_path in pdf_files:
            meta = _build_meta(
                pdf_path=pdf_path,
                prefix=args.doc_prefix,
                manifest_map=manifest_map,
                strict=args.strict,
            )
            one = ingester.ingest_document(pdf_path=pdf_path, meta=meta)
            results.append(one)

        ingested = sum(1 for r in results if r.get("status") == "ingested")
        skipped = len(results) - ingested

        print(
            json.dumps(
                {
                    "input": str(input_dir),
                    "recursive": args.recursive,
                    "total_pdfs": len(pdf_files),
                    "ingested": ingested,
                    "skipped": skipped,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
