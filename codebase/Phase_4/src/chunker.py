"""Chunk source text into verification-friendly evidence passages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class EvidenceChunk:
    chunk_id: str
    page_title: str
    page_url: str
    source_type: str
    retrieval_method: str
    text: str
    start_sentence_index: int
    end_sentence_index: int


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or " ").strip()
    if not text:
        return []
    # Keep it lightweight. Good enough for Wikipedia/external article prose.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def safe_title(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", (title or "page").replace(" ", "_"))[:80]


def build_chunks(
    page_title: str,
    page_url: str,
    text: str,
    source_type: str = "wikipedia",
    retrieval_method: str = "search",
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> List[EvidenceChunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    step = max(1, sentences_per_chunk - overlap_sentences)
    chunks: List[EvidenceChunk] = []
    base = safe_title(page_title)
    for start in range(0, len(sentences), step):
        end = min(len(sentences), start + sentences_per_chunk)
        chunk_text = " ".join(sentences[start:end]).strip()
        if len(chunk_text) < 45:
            continue
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{base}_{len(chunks)}",
                page_title=page_title,
                page_url=page_url,
                source_type=source_type,
                retrieval_method=retrieval_method,
                text=chunk_text,
                start_sentence_index=start,
                end_sentence_index=end,
            )
        )
        if end >= len(sentences):
            break
    return chunks
