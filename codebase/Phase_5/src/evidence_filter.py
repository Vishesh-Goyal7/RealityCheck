"""Evidence cleanup utilities for RealityCheck Phase 5.

These functions are intentionally conservative. Phase 5 should never mutate Phase 4
records, and it should never let bibliography/navigation junk create confident NLI
verdicts. Bad evidence + confident NLI = comedy with a lab coat.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

REFERENCE_PATTERNS = [
    r"\bISBN(?:-1[03])?\b",
    r"\bISSN\b",
    r"\bdoi\s*:\s*",
    r"\bBibcode\s*:\s*",
    r"\bPMID\b",
    r"\barXiv\b",
    r"\bRetrieved\b",
    r"\bArchived from the original\b",
    r"\bWayback Machine\b",
    r"\bReferences\b",
    r"\bBibliography\b",
    r"\bFurther reading\b",
    r"\bExternal links\b",
    r"\bOfficial website\b",
    r"\bISBN\s*97[89]",
    r"\b\d+\s*\(\d+\)\s*:\s*\d+",
    r"\bpp?\.\s*\d+",
    r"\bvol\.\s*\d+",
]

SECTION_JUNK_PATTERNS = [
    r"==\s*(References|Bibliography|Further reading|External links|See also|Notes)\s*==",
    r"===\s*(Official|Other|Bibliography|Politics|Lists)\s*===",
    r"Official website of",
    r"Column archive at",
    r"\bList of .* endorsements\b",
    r"\bList of presidents of the United States\b",
    r"\bSee also\b.*\bReferences\b",
]

BIBLIOGRAPHIC_SHAPE = [
    # Kruszelnicki, Karl; Adam Yazxhi (2006). Great Mythconceptions...
    r"^[A-Z][A-Za-z'\-]+,\s+[A-Z][A-Za-z'\-]+.*\(\d{4}\)\.",
    r"^[A-Z][A-Za-z'\-]+\s+[A-Z]\..*\(\d{4}\)",
    r"\b(ed\.|eds\.|Publishing|Press|Journal|Historian|Nature|Science)\b",
]


def looks_like_reference_chunk(text: str) -> bool:
    """Return True if a chunk looks like bibliography/citation/navigation junk."""
    if not text or not text.strip():
        return True

    stripped = re.sub(r"\s+", " ", text.strip())
    lower = stripped.lower()
    words = stripped.split()

    if len(words) < 7:
        return True

    # Section/list/navigation dumps from Wikipedia articles.
    if stripped.count("==") >= 1 and any(re.search(p, stripped, re.I) for p in SECTION_JUNK_PATTERNS):
        return True

    # Heavy heading/list structure is almost never useful evidence.
    if stripped.count("==") >= 2:
        return True

    ref_hits = sum(1 for pat in REFERENCE_PATTERNS if re.search(pat, stripped, flags=re.I))
    biblio_hits = sum(1 for pat in BIBLIOGRAPHIC_SHAPE if re.search(pat, stripped, flags=re.I))

    if ref_hits >= 2:
        return True
    if biblio_hits >= 1 and len(words) < 45:
        return True
    if biblio_hits >= 2 and len(words) < 100:
        return True

    # Single citation-like sentence with date and publisher/journal is junk.
    if re.search(r"\(\d{4}\)\.", stripped) and re.search(r"\b(Publishing|Press|Journal|Nature|Science|Historian)\b", stripped):
        return True

    if lower.count("official website") >= 1 and len(words) > 25:
        return True

    # Long list of unrelated links/items.
    semicolon_count = stripped.count(";")
    if semicolon_count >= 5 and len(words) > 60:
        return True

    alpha_words = re.findall(r"[A-Za-z]{4,}", stripped)
    if len(alpha_words) < 5:
        return True

    return False


def looks_like_generic_explainer_chunk(text: str) -> bool:
    """Detect source-page boilerplate that explains what a list/page is, not the claim."""
    stripped = re.sub(r"\s+", " ", (text or "").strip()).lower()
    generic_markers = [
        "each entry on these lists",
        "each entry on this list",
        "these entries are concise summaries",
        "readers can consult the main subject articles",
        "common misconceptions are widely accepted",
        "the misconceptions themselves are implied",
    ]
    return any(marker in stripped for marker in generic_markers)


def filter_evidence_chunks(
    chunks: List[Dict[str, Any]],
    *,
    max_chunks: int = 5,
    min_adjusted_similarity: float = 0.52,
    keep_at_least: int = 2,
) -> List[Dict[str, Any]]:
    """Filter and sort Phase 4 evidence chunks for NLI verification.

    Conservative behavior:
    - remove obvious bibliography/navigation junk
    - keep source chunks slightly more generously
    - if filtering is too aggressive, keep the best non-empty non-junk chunks as fallback
    """
    raw_chunks = list(chunks or [])
    usable: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []

    for chunk in raw_chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        score = chunk.get("adjusted_similarity_score", chunk.get("similarity_score", 0.0))
        try:
            score = float(score)
        except Exception:
            score = 0.0

        copied = dict(chunk)
        copied["phase5_usable_score"] = score

        is_junk = looks_like_reference_chunk(text)
        is_generic_explainer = looks_like_generic_explainer_chunk(text)
        if is_junk:
            copied.setdefault("phase5_flags", [])
            copied["phase5_flags"] = list(copied["phase5_flags"]) + ["filtered_reference_or_navigation_junk"]
            continue

        if is_generic_explainer:
            copied.setdefault("phase5_flags", [])
            copied["phase5_flags"] = list(copied["phase5_flags"]) + ["generic_page_explainer_not_direct_evidence"]
            # Keep as low-priority context only; it must not become sole proof.
            copied["phase5_usable_score"] = min(score, 0.50)

        fallback.append(copied)

        flags = chunk.get("chunk_quality_flags", []) or []
        is_source_chunk = "source_chunk" in flags
        threshold = min_adjusted_similarity - 0.05 if is_source_chunk else min_adjusted_similarity
        effective_score = float(copied.get("phase5_usable_score", score) or 0.0)

        if effective_score < threshold:
            continue

        usable.append(copied)

    usable.sort(key=lambda c: float(c.get("phase5_usable_score", 0.0)), reverse=True)
    fallback.sort(key=lambda c: float(c.get("phase5_usable_score", 0.0)), reverse=True)

    if not usable and fallback:
        out = []
        for c in fallback[: max(1, min(max_chunks, keep_at_least))]:
            c = dict(c)
            c.setdefault("phase5_flags", [])
            c["phase5_flags"] = list(c["phase5_flags"]) + ["fallback_kept_after_filter"]
            out.append(c)
        return out

    return usable[:max_chunks]
