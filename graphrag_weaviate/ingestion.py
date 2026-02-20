from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import Chunk, ChunkType, Document, Recommendation, Section
from .storage import WeaviateGraphStore
from .utils import estimate_tokens, load_json, normalize_space, save_json, stable_hash

log = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^\s*((?:\d+\.){0,4}\d+)\s+(.+)$")
BLOCK_START_RE = re.compile(
    r"\b(рекомендац|алгоритм|таблица|критери|приложени|recommendation|algorithm|table)\b",
    re.IGNORECASE,
)


class IngestionService:
    def __init__(self, store: WeaviateGraphStore, checkpoint_file: str = ".graphrag_ingest_checkpoint.json"):
        self.store = store
        self.checkpoint_path = Path(checkpoint_file)

    def ingest(self, input_dir: str, manifest_path: str) -> dict[str, Any]:
        started = time.time()
        log.info("Ingestion started input_dir=%s manifest=%s", input_dir, manifest_path)
        manifest = self._load_manifest(Path(manifest_path))
        checkpoint = load_json(self.checkpoint_path)
        summary: dict[str, Any] = {"docs_total": 0, "docs_ingested": 0, "docs_skipped": 0, "docs": []}
        input_dir_path = Path(input_dir)
        used_pdf_names: set[str] = set()

        for manifest_key, spec in sorted(manifest.items(), key=lambda kv: str(kv[1].get("doc_id") or kv[0])):
            doc_id = str(spec.get("doc_id") or "").strip()
            if not doc_id:
                log.warning("Skipping manifest record %s: empty doc_id", manifest_key)
                continue
            pdf_file = self._resolve_pdf_file(input_dir_path, manifest_key, spec)
            if not pdf_file:
                log.warning("Skipping doc_id=%s: no matching PDF found in %s", doc_id, input_dir)
                continue
            used_pdf_names.add(pdf_file.name)
            summary["docs_total"] += 1
            if checkpoint.get(doc_id) == "done":
                summary["docs_skipped"] += 1
                log.info("Skipping doc_id=%s: already marked done in checkpoint", doc_id)
                continue
            one = self.ingest_document(pdf_file, spec)
            summary["docs"].append(one)
            if one["status"] == "ingested":
                summary["docs_ingested"] += 1
                checkpoint[doc_id] = "done"
                save_json(self.checkpoint_path, checkpoint)
            else:
                summary["docs_skipped"] += 1

        for pdf_file in sorted(input_dir_path.glob("*.pdf")):
            if pdf_file.name not in used_pdf_names:
                log.warning("Ignoring %s: no manifest record", pdf_file.name)

        summary["runtime_sec"] = round(time.time() - started, 3)
        log.info(
            "Ingestion finished docs_total=%d docs_ingested=%d docs_skipped=%d runtime_sec=%.3f",
            summary["docs_total"],
            summary["docs_ingested"],
            summary["docs_skipped"],
            summary["runtime_sec"],
        )
        return summary

    def _resolve_pdf_file(self, input_dir: Path, manifest_key: str, spec: dict[str, Any]) -> Path | None:
        candidates: list[str] = []
        doc_id = str(spec.get("doc_id") or "").strip()
        if doc_id:
            candidates.append(f"{doc_id}.pdf")
        explicit = str(spec.get("filename") or spec.get("file") or "").strip()
        if explicit:
            candidates.append(explicit)
        key = str(manifest_key).strip()
        if key:
            candidates.append(key)
            if not key.lower().endswith(".pdf"):
                candidates.append(f"{key}.pdf")

        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            path = input_dir / normalized
            if path.is_file() and path.suffix.lower() == ".pdf":
                return path
        return None

    def ingest_document(self, pdf_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time()
        doc_id = str(meta["doc_id"])
        title = str(meta.get("title") or pdf_path.stem)
        log.info("Ingest document started doc_id=%s file=%s", doc_id, pdf_path)
        pages = self._extract_pages(pdf_path)
        doc_hash = stable_hash("\n".join([p["text"] for p in pages]))

        if self.store.find_document_by_hash(doc_hash):
            log.info("Skip duplicate document hash doc_id=%s", doc_id)
            return {
                "doc_id": doc_id,
                "status": "skipped_duplicate_document_hash",
                "pages": len(pages),
                "chunks": 0,
                "runtime_sec": round(time.time() - t0, 3),
            }

        doc = Document(
            doc_id=doc_id,
            title=title,
            year=meta.get("year"),
            specialty=meta.get("specialty"),
            source_url=meta.get("source_url"),
            metadata_json=json.dumps(meta, ensure_ascii=False),
            hash=doc_hash,
        )
        self.store.upsert_document(doc)

        sections = self._detect_sections(pdf_path, pages, doc_id)
        section_lookup: dict[str, str] = {}
        for section in sections:
            section_id = self.store.upsert_section(section)
            section_lookup[section.path] = section_id

        chunks = self._chunk_sections(pages, sections, doc_id)
        total_tokens = 0
        for idx, chunk in enumerate(chunks):
            chunk.order = idx
            chunk.section_path = chunk.section_path or "document"
            chunk.token_count = chunk.token_count or estimate_tokens(chunk.chunk_text)
            total_tokens += chunk.token_count
            embedding = self.store.embed_text(chunk.chunk_text)
            chunk_id = self.store.upsert_chunk(chunk, embedding=embedding)
            section_id = section_lookup.get(chunk.section_path)
            if section_id:
                self.store.link_chunk_to_section(chunk_id, section_id)
            self.store.link_chunk_to_document(chunk_id, doc_id)
            if chunk.chunk_type in {ChunkType.RECOMMENDATION, ChunkType.ALGORITHM}:
                rec = Recommendation(statement=chunk.chunk_text)
                rec_id = self.store.upsert_recommendation(rec, doc_id=doc_id)
                self.store.link_recommendation_to_chunk(rec_id, chunk_id)

        result = {
            "doc_id": doc_id,
            "status": "ingested",
            "pages": len(pages),
            "sections": len(sections),
            "chunks": len(chunks),
            "avg_tokens": round(total_tokens / max(1, len(chunks)), 2),
            "runtime_sec": round(time.time() - t0, 3),
        }
        log.info(
            "Ingest document finished doc_id=%s pages=%d sections=%d chunks=%d runtime_sec=%.3f",
            doc_id,
            result["pages"],
            result["sections"],
            result["chunks"],
            result["runtime_sec"],
        )
        return result

    def _find_manifest_spec(self, pdf_file: Path, manifest: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        direct = manifest.get(pdf_file.name) or manifest.get(pdf_file.stem)
        if direct:
            return direct
        for spec in manifest.values():
            if str(spec.get("doc_id") or "").strip() == pdf_file.stem:
                return spec
        return None

    def _normalize_manifest_row(self, row: dict[str, Any], fallback_doc_id: str | None = None) -> dict[str, Any]:
        doc_id = str(row.get("doc_id") or row.get("ID") or row.get("id") or fallback_doc_id or "").strip()
        title = row.get("title") or row.get("Наименование")
        year = self._to_int_or_none(row.get("year"))
        specialty = row.get("specialty")
        source_url = row.get("source_url")

        out = dict(row)
        out["doc_id"] = doc_id
        if title:
            out["title"] = title
        if year:
            out["year"] = year
        if specialty:
            out["specialty"] = specialty
        if source_url:
            out["source_url"] = source_url
        return out

    def _to_int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.isdigit():
                return int(stripped)
            return None
        return None

    def _load_manifest_from_json(self, data: Any) -> dict[str, dict[str, Any]]:
        if isinstance(data, list):
            out: dict[str, dict[str, Any]] = {}
            for row in data:
                filename = str(row.get("filename") or row.get("file") or "").strip()
                if not filename:
                    continue
                out[filename] = self._normalize_manifest_row(row, fallback_doc_id=Path(filename).stem)
            return out
        if isinstance(data, dict):
            if "documents" in data and isinstance(data["documents"], list):
                out: dict[str, dict[str, Any]] = {}
                for row in data["documents"]:
                    filename = str(row.get("filename") or row.get("file") or "").strip()
                    if not filename:
                        continue
                    out[filename] = self._normalize_manifest_row(row, fallback_doc_id=Path(filename).stem)
                return out
            out: dict[str, dict[str, Any]] = {}
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                key_s = str(key).strip()
                out[key_s] = self._normalize_manifest_row(value, fallback_doc_id=Path(key_s).stem)
            return out
        raise ValueError("Unsupported manifest JSON format")

    def _load_manifest_from_csv(self, path: Path) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {str(k).strip(): v for k, v in row.items() if k is not None}
                filename = str(row.get("filename") or row.get("file") or "").strip()
                doc_id = str(row.get("doc_id") or row.get("ID") or row.get("id") or "").strip()
                if not filename and doc_id:
                    filename = f"{doc_id}.pdf"
                if not filename:
                    continue
                out[filename] = self._normalize_manifest_row(row, fallback_doc_id=Path(filename).stem)
        return out

    def _load_manifest(self, path: Path) -> dict[str, dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            return self._load_manifest_from_csv(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        return self._load_manifest_from_json(data)

    def _extract_pages(self, pdf_path: Path) -> list[dict[str, Any]]:
        import fitz

        doc = fitz.open(pdf_path)
        pages: list[dict[str, Any]] = []
        for i, page in enumerate(doc):
            text = normalize_space(page.get_text("text"))
            pages.append({"page": i + 1, "text": text})
        doc.close()
        return pages

    def _detect_sections(self, pdf_path: Path, pages: list[dict[str, Any]], doc_id: str) -> list[Section]:
        by_page_heading = self._detect_headings_from_text(pages)
        toc_sections = self._detect_from_toc(pdf_path)

        if toc_sections:
            sections = []
            for i, item in enumerate(toc_sections):
                page_start = item["page"]
                page_end = toc_sections[i + 1]["page"] - 1 if i + 1 < len(toc_sections) else pages[-1]["page"]
                sections.append(
                    Section(
                        doc_id=doc_id,
                        path=item["path"],
                        order=i,
                        level=item["level"],
                        page_start=page_start,
                        page_end=max(page_start, page_end),
                    )
                )
            return sections

        if by_page_heading:
            sections = []
            for i, item in enumerate(by_page_heading):
                page_start = item["page"]
                page_end = by_page_heading[i + 1]["page"] - 1 if i + 1 < len(by_page_heading) else pages[-1]["page"]
                sections.append(
                    Section(
                        doc_id=doc_id,
                        path=item["path"],
                        order=i,
                        level=item["level"],
                        page_start=page_start,
                        page_end=max(page_start, page_end),
                    )
                )
            return sections

        return [
            Section(
                doc_id=doc_id,
                path="document",
                order=0,
                level=1,
                page_start=1,
                page_end=pages[-1]["page"],
            )
        ]

    def _detect_from_toc(self, pdf_path: Path) -> list[dict[str, Any]]:
        import fitz

        doc = fitz.open(pdf_path)
        toc = doc.get_toc(simple=True)
        doc.close()
        if not toc:
            return []
        items = []
        for entry in toc:
            level, title, page = entry
            items.append(
                {
                    "level": int(level),
                    "path": normalize_space(str(title)),
                    "page": int(page) if page else 1,
                }
            )
        return items

    def _detect_headings_from_text(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in pages:
            lines = p["text"].split(". ")
            for line in lines[:6]:
                m = HEADING_RE.match(line.strip())
                if not m:
                    continue
                section_code = m.group(1)
                title = m.group(2)[:200]
                path = f"{section_code} {title}".strip()
                if path in seen:
                    continue
                seen.add(path)
                level = section_code.count(".") + 1
                items.append({"level": level, "path": path, "page": p["page"]})
        items.sort(key=lambda x: x["page"])
        return items

    def _chunk_sections(self, pages: list[dict[str, Any]], sections: list[Section], doc_id: str) -> list[Chunk]:
        text_by_page = {p["page"]: p["text"] for p in pages}
        chunks: list[Chunk] = []

        for section in sections:
            if section.page_start is None or section.page_end is None:
                continue
            section_pages = list(range(section.page_start, section.page_end + 1))
            section_text = "\n\n".join(text_by_page.get(p, "") for p in section_pages).strip()
            if not section_text:
                continue
            section_chunks = self._split_into_chunks(section_text)
            for text in section_chunks:
                text = normalize_space(text)
                if not text:
                    continue
                ctype = self._classify_chunk_type(text, section.path)
                chunk_hash = stable_hash(f"{doc_id}|{section.path}|{text}")
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        section_path=section.path,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        chunk_text=text,
                        chunk_type=ctype,
                        token_count=estimate_tokens(text),
                        chunk_hash=chunk_hash,
                        entity_mentions=self._extract_entities(text),
                    )
                )
        return chunks

    def _split_into_chunks(self, text: str, min_tokens: int = 500, max_tokens: int = 1200) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        blocks: list[str] = []
        carry = []
        for para in paragraphs:
            if BLOCK_START_RE.search(para) and carry:
                blocks.append("\n\n".join(carry).strip())
                carry = [para]
            else:
                carry.append(para)
        if carry:
            blocks.append("\n\n".join(carry).strip())

        chunks: list[str] = []
        current: list[str] = []
        tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block)
            if block_tokens > max_tokens and BLOCK_START_RE.search(block):
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    tokens = 0
                chunks.append(block)
                continue
            if tokens + block_tokens > max_tokens and tokens >= min_tokens:
                chunks.append("\n\n".join(current))
                current = [block]
                tokens = block_tokens
            else:
                current.append(block)
                tokens += block_tokens

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _classify_chunk_type(self, text: str, section_path: str) -> ChunkType:
        source = f"{section_path} {text[:300]}".lower()
        if "рекомендац" in source or "recommend" in source:
            return ChunkType.RECOMMENDATION
        if "алгоритм" in source or "algorithm" in source:
            return ChunkType.ALGORITHM
        if "таблица" in source or "table" in source:
            return ChunkType.TABLE
        if "определен" in source or "definition" in source:
            return ChunkType.DEFINITION
        if "доказатель" in source or "evidence" in source:
            return ChunkType.EVIDENCE
        if "приложени" in source or "appendix" in source:
            return ChunkType.APPENDIX
        return ChunkType.OTHER

    def _extract_entities(self, text: str) -> list[str]:
        # Simple heuristic, can be replaced with NER model.
        terms = re.findall(r"\b[А-ЯA-Z][а-яa-zA-Z\-]{3,}\b", text)
        freq: dict[str, int] = defaultdict(int)
        for term in terms:
            freq[term] += 1
        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        return [w for w, _ in ranked[:10]]
