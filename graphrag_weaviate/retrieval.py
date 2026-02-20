from __future__ import annotations

import logging
import re
from collections import defaultdict

from .config import Settings
from .models import ChunkRecord, ChunkType
from .storage import WeaviateGraphStore

log = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, store: WeaviateGraphStore, settings: Settings):
        self.store = store
        self.settings = settings

    def rerank(self, query: str, candidates: list[ChunkRecord]) -> list[ChunkRecord]:
        query_terms = self._terms(query)
        for cand in candidates:
            text_terms = self._terms(cand.chunk_text)
            overlap = len(query_terms.intersection(text_terms))
            type_boost = 0.2 if cand.chunk_type in {ChunkType.RECOMMENDATION, ChunkType.ALGORITHM} else 0.0
            cand.score = cand.score + overlap * 0.05 + type_boost
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def expand_graph(self, seed_chunk_ids: list[str], doc_id: str, budget: int) -> list[str]:
        seeds = self.store.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=seed_chunk_ids)
        expanded: set[str] = set()
        remaining = budget

        for seed in seeds:
            if remaining <= 0:
                break
            neighbors = self.store.fetch_section_neighbors(
                doc_id=doc_id,
                section_path=seed.section_path,
                center_order=seed.order,
                limit=min(3, remaining),
            )
            for n in neighbors:
                if n.chunk_id not in seed_chunk_ids and n.chunk_id not in expanded:
                    expanded.add(n.chunk_id)
                    remaining -= 1
                    if remaining <= 0:
                        break

        if remaining > 0:
            entity_terms = []
            for seed in seeds:
                entity_terms.extend(self._extract_terms_for_expansion(seed.chunk_text)[:3])
            by_entity = self.store.fetch_chunks_by_entity_mentions(doc_id=doc_id, entities=entity_terms, limit=remaining)
            for n in by_entity:
                if n.chunk_id not in seed_chunk_ids and n.chunk_id not in expanded:
                    expanded.add(n.chunk_id)
                    remaining -= 1
                    if remaining <= 0:
                        break

        if remaining > 0:
            rec_related = self.store.fetch_chunks_supported_by_recommendations(
                doc_id=doc_id,
                chunk_ids=seed_chunk_ids,
                limit=remaining,
            )
            for n in rec_related:
                if n.chunk_id not in seed_chunk_ids and n.chunk_id not in expanded:
                    expanded.add(n.chunk_id)
                    remaining -= 1
                    if remaining <= 0:
                        break

        return list(expanded)

    def pack_context(self, query: str, chunk_records: list[ChunkRecord], target_n: int = 8) -> list[ChunkRecord]:
        target_n = max(self.settings.packed_min, min(self.settings.packed_max, target_n))
        query_terms = self._terms(query)

        scored = []
        seen_text = set()
        for c in chunk_records:
            norm = re.sub(r"\s+", " ", c.chunk_text.strip().lower())
            if norm in seen_text:
                continue
            seen_text.add(norm)
            lexical = len(query_terms.intersection(self._terms(c.chunk_text)))
            type_priority = 2 if c.chunk_type in {ChunkType.RECOMMENDATION, ChunkType.ALGORITHM} else 0
            scored.append((type_priority, lexical, c.score, c))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        packed: list[ChunkRecord] = []
        section_counter: dict[str, int] = defaultdict(int)
        for _, _, _, c in scored:
            if len(packed) >= target_n:
                break
            # minimal redundancy and section diversity
            if section_counter[c.section_path] >= 3 and len(packed) < target_n - 1:
                continue
            packed.append(c)
            section_counter[c.section_path] += 1

        packed.sort(key=lambda x: (x.page_start, x.order))
        return packed

    def retrieve_context(
        self,
        doc_id: str,
        query: str,
    ) -> list[ChunkRecord]:
        log.info("Retrieve context start doc_id=%s query_len=%d", doc_id, len(query))
        candidates = self.store.hybrid_search_chunks(
            doc_id=doc_id,
            query=query,
            limit=self.settings.k_initial,
        )[: self.settings.k_initial]
        log.info("Retrieve context candidates=%d", len(candidates))

        ranked = self.rerank(query=query, candidates=candidates)[: self.settings.k_top]
        log.info("Retrieve context reranked_top=%d", len(ranked))
        seed_ids = [r.chunk_id for r in ranked]
        expanded_ids = self.expand_graph(seed_chunk_ids=seed_ids, doc_id=doc_id, budget=self.settings.k_expand)
        log.info("Retrieve context expanded=%d", len(expanded_ids))

        expanded_records = self.store.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=expanded_ids)
        all_records = ranked + expanded_records

        packed = self.pack_context(query=query, chunk_records=all_records, target_n=self.settings.packed_max)
        log.info("Retrieve context packed=%d", len(packed))
        return packed

    def _terms(self, text: str) -> set[str]:
        return set(re.findall(r"[\w\-]{3,}", text.lower()))

    def _extract_terms_for_expansion(self, text: str) -> list[str]:
        words = re.findall(r"\b[А-ЯA-Z][а-яa-zA-Z\-]{3,}\b", text)
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
        return seen
