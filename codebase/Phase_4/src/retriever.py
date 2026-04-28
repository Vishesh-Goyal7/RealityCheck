"""Final source-aware evidence retrieval engine for RealityCheck Phase 4."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from chunker import EvidenceChunk, build_chunks
from embedder import EvidenceEmbedder
from query_builder import build_ranking_query, build_search_queries, compact_text, extract_key_terms_for_overlap
from wiki_client import EvidencePage, PSEUDO_SOURCE_VALUES, SourceAwareClient


class EvidenceRetriever:
    def __init__(
        self,
        source_client: SourceAwareClient | None = None,
        embedder: EvidenceEmbedder | None = None,
        top_pages: int = 3,
        candidate_pages: int = 8,
        top_chunks: int = 5,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1,
        min_chunk_similarity: float = 0.50,
        min_page_similarity: float = 0.15,
        source_page_bonus: float = 0.20,
        source_chunk_bonus: float = 0.06,
    ) -> None:
        self.source_client = source_client or SourceAwareClient()
        self.embedder = embedder or EvidenceEmbedder()
        self.top_pages = top_pages
        self.candidate_pages = max(candidate_pages, top_pages)
        self.top_chunks = top_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.min_chunk_similarity = min_chunk_similarity
        self.min_page_similarity = min_page_similarity
        self.source_page_bonus = source_page_bonus
        self.source_chunk_bonus = source_chunk_bonus

    def retrieve_for_claim(self, record: Dict[str, Any], claim: Dict[str, Any], source_url: str | None = None) -> Dict[str, Any]:
        if not claim.get("is_checkable", False):
            return self._skipped_result(claim, "skipped_non_checkable")

        ranking_query = build_ranking_query(record, claim)
        search_queries = build_search_queries(record, claim, source_url=source_url)
        if not ranking_query and not search_queries:
            return self._skipped_result(claim, "no_query")

        pages = self.source_client.retrieve_pages(
            queries=search_queries,
            limit_per_query=3,
            max_pages=self.candidate_pages,
            source_url=source_url,
            extra_text=f"{record.get('question') or ''} {claim.get('sentence_text') or ''} {claim.get('normalized_text') or ''}",
        )

        source_attempted = bool(source_url and str(source_url).strip().lower() not in PSEUDO_SOURCE_VALUES)
        ranked_pages = self._rank_pages(ranking_query, pages)
        selected_pages = self._select_pages(ranked_pages)

        all_chunks: List[EvidenceChunk] = []
        for page in selected_pages:
            all_chunks.extend(
                build_chunks(
                    page_title=page.title,
                    page_url=page.url,
                    text=page.extract,
                    source_type=page.source_type,
                    retrieval_method=page.retrieval_method,
                    sentences_per_chunk=self.sentences_per_chunk,
                    overlap_sentences=self.overlap_sentences,
                )
            )

        evidence_chunks = self._rank_chunks(ranking_query, record, claim, all_chunks)
        status, quality_flags = self._status_and_flags(
            evidence_chunks=evidence_chunks,
            selected_pages=selected_pages,
            ranked_pages=ranked_pages,
            source_attempted=source_attempted,
            source_url=source_url,
        )

        return {
            "claim_id": claim.get("claim_id"),
            "sentence_text": claim.get("sentence_text"),
            "normalized_text": claim.get("normalized_text"),
            "importance": claim.get("importance"),
            "is_checkable": True,
            "retrieval_status": status,
            "retrieval_quality_flags": quality_flags,
            "search_queries_used": search_queries,
            "ranking_query": ranking_query,
            "source_url_used": source_url or "",
            "retrieval_debug": getattr(self.source_client, "debug_events", []),
            "retrieved_pages": [
                {
                    "page_title": page.title,
                    "page_url": page.url,
                    "page_id": page.page_id,
                    "retrieval_method": page.retrieval_method,
                    "source_type": page.source_type,
                    "page_similarity_score": score,
                }
                for page, score in ranked_pages[: self.candidate_pages]
            ],
            "selected_pages": [
                {
                    "page_title": page.title,
                    "page_url": page.url,
                    "page_id": page.page_id,
                    "retrieval_method": page.retrieval_method,
                    "source_type": page.source_type,
                }
                for page in selected_pages
            ],
            "evidence_chunks": evidence_chunks,
        }

    def retrieve_for_record(self, record: Dict[str, Any], source_url: str | None = None) -> Dict[str, Any]:
        claims = record.get("claims", [])
        evidence_results = [self.retrieve_for_claim(record, claim, source_url=source_url) for claim in claims]
        return {
            "id": record.get("id"),
            "question": record.get("question"),
            "model_name": record.get("model_name"),
            "model_id": record.get("model_id"),
            "response_quality": record.get("response_quality"),
            "source": source_url or record.get("source", ""),
            "cleaned_response": record.get("cleaned_response"),
            "claims": claims,
            "evidence_results": evidence_results,
            "checkable_claim_count": record.get("checkable_claim_count", 0),
            "non_checkable_claim_count": record.get("non_checkable_claim_count", 0),
        }

    # ----------------------------- ranking -----------------------------

    def _rank_pages(self, ranking_query: str, pages: List[EvidencePage]) -> List[Tuple[EvidencePage, float]]:
        if not pages:
            return []
        page_texts = []
        for page in pages:
            intro = compact_text((page.extract or "")[:2200])
            page_texts.append(f"{page.title}. {intro}")
        ranked = self.embedder.rank(ranking_query, page_texts, top_k=len(page_texts))
        result = []
        for idx, score in ranked:
            page = pages[idx]
            adjusted = float(score)
            if page.retrieval_method in {"source_url", "external_source"}:
                adjusted += self.source_page_bonus
            result.append((page, adjusted))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def _select_pages(self, ranked_pages: List[Tuple[EvidencePage, float]]) -> List[EvidencePage]:
        if not ranked_pages:
            return []
        selected: List[EvidencePage] = []

        # Source pages are privileged: they are the benchmark's intended evidence route.
        for page, _ in ranked_pages:
            if page.retrieval_method in {"source_url", "external_source"} and page.extract.strip():
                selected.append(page)

        # Add best search pages above a loose threshold.
        for page, score in ranked_pages:
            if len(selected) >= self.top_pages:
                break
            if page in selected:
                continue
            if score >= self.min_page_similarity:
                selected.append(page)

        # If every page is below threshold but exists, keep the top page as weak evidence candidate.
        if not selected and ranked_pages:
            selected.append(ranked_pages[0][0])

        return selected[: self.top_pages]

    def _rank_chunks(self, ranking_query: str, record: Dict, claim: Dict, chunks: List[EvidenceChunk]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        terms = extract_key_terms_for_overlap(record, claim)
        ranked = self.embedder.rank(ranking_query, [c.text for c in chunks], top_k=min(len(chunks), max(self.top_chunks * 5, self.top_chunks)))

        scored: List[Tuple[EvidenceChunk, float, float, List[str]]] = []
        for idx, raw_score in ranked:
            chunk = chunks[idx]
            if self._is_obviously_noisy_chunk(chunk.text):
                continue
            adjusted = float(raw_score)
            flags: List[str] = []
            if chunk.retrieval_method in {"source_url", "external_source"}:
                adjusted += self.source_chunk_bonus
                flags.append("source_chunk")
            overlap = self._term_overlap_count(chunk.text, terms)
            if overlap >= 2:
                adjusted += 0.04
                flags.append("term_overlap_boost")
            if adjusted < self.min_chunk_similarity:
                continue
            scored.append((chunk, float(raw_score), adjusted, flags))

        scored.sort(key=lambda x: x[2], reverse=True)
        evidence = []
        for chunk, raw_score, adjusted, flags in scored[: self.top_chunks]:
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "chunk_id": chunk.chunk_id,
                    "page_title": chunk.page_title,
                    "page_url": chunk.page_url,
                    "source_type": chunk.source_type,
                    "retrieval_method": chunk.retrieval_method,
                    "text": chunk.text,
                    "similarity_score": raw_score,
                    "adjusted_similarity_score": adjusted,
                    "chunk_quality_flags": flags,
                    "start_sentence_index": chunk.start_sentence_index,
                    "end_sentence_index": chunk.end_sentence_index,
                }
            )
        return evidence

    # ----------------------------- quality -----------------------------

    def _status_and_flags(
        self,
        evidence_chunks: List[Dict[str, Any]],
        selected_pages: List[EvidencePage],
        ranked_pages: List[Tuple[EvidencePage, float]],
        source_attempted: bool,
        source_url: str | None,
    ) -> tuple[str, List[str]]:
        flags: List[str] = []
        source_selected = any(p.retrieval_method in {"source_url", "external_source"} for p in selected_pages)
        external_selected = any(p.source_type == "external" for p in selected_pages)

        if source_attempted and not source_selected:
            flags.append("source_unavailable_or_unselected")
        if source_selected:
            flags.append("source_page_used")
        if external_selected:
            flags.append("external_source_used")

        if not ranked_pages:
            return "no_evidence_found", flags + ["no_pages_retrieved"]
        if not selected_pages:
            return "no_evidence_found", flags + ["no_pages_selected"]
        if not evidence_chunks:
            return "no_evidence_found", flags + ["no_chunks_passed_filter"]

        top_adjusted = evidence_chunks[0].get("adjusted_similarity_score", evidence_chunks[0].get("similarity_score", 0.0))
        if top_adjusted < 0.58:
            flags.append("weak_chunk_match")
        if not source_selected and ranked_pages[0][1] < 0.42:
            flags.append("weak_page_match")

        weak = any(f in flags for f in ["weak_chunk_match", "weak_page_match"])
        return ("weak_evidence" if weak else "ok"), flags

    @staticmethod
    def _term_overlap_count(text: str, terms: List[str]) -> int:
        low = text.lower()
        count = 0
        for term in terms:
            t = term.lower().strip()
            if len(t) >= 4 and t in low:
                count += 1
        return count

    @staticmethod
    def _is_obviously_noisy_chunk(text: str) -> bool:
        t = re.sub(r"\s+", " ", text or "").strip()
        if len(t) < 45:
            return True
        bad_patterns = [
            r"\bISBN\b", r"\bOCLC\b", r"\bdoi:", r"\bISSN\b", r"Retrieved \d{1,2}",
            r"Archived from the original", r"Wayback Machine", r"\bpp\.\s*\d+", r"\bVol\.\b",
            r"Project Gutenberg", r"LibriVox", r"at IMDb", r"External links"
        ]
        hits = sum(1 for p in bad_patterns if re.search(p, t, flags=re.IGNORECASE))
        if hits >= 2:
            return True
        # citation/reference-like chunks with too many years and too little prose
        years = len(re.findall(r"\b(?:18|19|20)\d{2}\b", t))
        if years >= 4 and len(t) < 350:
            return True
        if len(re.findall(r"[.!?]", t)) <= 1 and hits >= 1:
            return True
        return False

    @staticmethod
    def _skipped_result(claim: Dict[str, Any], status: str) -> Dict[str, Any]:
        return {
            "claim_id": claim.get("claim_id"),
            "sentence_text": claim.get("sentence_text"),
            "is_checkable": claim.get("is_checkable", False),
            "retrieval_status": status,
            "retrieval_quality_flags": [],
            "evidence_query": claim.get("evidence_query", ""),
            "retrieved_pages": [],
            "selected_pages": [],
            "evidence_chunks": [],
        }
