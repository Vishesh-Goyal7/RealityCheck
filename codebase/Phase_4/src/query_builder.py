"""Query construction utilities for source-aware Phase 4 retrieval."""

from __future__ import annotations

import re
from typing import Dict, List

STOPWORDS = {
    "what", "who", "why", "when", "where", "how", "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "from", "with", "and", "or", "but", "as",
    "it", "this", "that", "there", "they", "them", "he", "she", "his", "her", "their", "your", "you", "i",
    "yes", "no", "not", "actually", "really", "usually", "generally", "likely", "because", "about",
    "single", "factual", "answer", "question", "people", "thing", "things", "said", "called", "has", "have",
    "had", "been", "being", "will", "would", "could", "should", "may", "might", "can", "cannot"
}


def compact_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"[^A-Za-z0-9'\"\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, max_terms: int = 10) -> str:
    text = compact_text(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*|\d{4}|\d+", text)
    out: List[str] = []
    seen = set()
    for tok in tokens:
        low = tok.lower().strip("'\"")
        if len(low) <= 2 and not low.isdigit():
            continue
        if low in STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(tok.strip("'\""))
        if len(out) >= max_terms:
            break
    return " ".join(out)


def extract_quoted_phrases(text: str) -> List[str]:
    phrases = re.findall(r"[\"“”']([^\"“”']{4,140})[\"“”']", text or "")
    return [compact_text(p) for p in phrases if compact_text(p)]


def title_from_source_url(source_url: str | None) -> str:
    if not source_url or "/wiki/" not in source_url:
        return ""
    raw = source_url.split("/wiki/", 1)[-1].split("#", 1)[0].split("?", 1)[0]
    raw = raw.replace("_", " ")
    return compact_text(raw)


def build_search_queries(record: Dict, claim: Dict, source_url: str | None = None, max_queries: int = 6) -> List[str]:
    question = record.get("question") or ""
    sentence = claim.get("sentence_text") or ""
    category = (claim.get("metadata", {}) or {}).get("category", "")

    queries: List[str] = []

    # Exact phrases first, especially for misquotations.
    for phrase in extract_quoted_phrases(question + " " + sentence):
        queries.append(phrase)

    # If the source URL itself is a wiki page, its title is a high quality query.
    source_title = title_from_source_url(source_url)
    if source_title:
        queries.append(source_title)

    q_kw = extract_keywords(question, max_terms=8)
    s_kw = extract_keywords(sentence, max_terms=8)
    combined_kw = extract_keywords(question + " " + sentence, max_terms=12)

    if q_kw:
        queries.append(q_kw)
    if combined_kw:
        queries.append(combined_kw)
    if s_kw:
        queries.append(s_kw)

    cat_low = category.lower()
    if "misquotation" in cat_low:
        queries.append("common misquotations")
    if "misconception" in cat_low:
        queries.append("common misconceptions")
    if "conspirac" in cat_low and q_kw:
        queries.append("conspiracy theories " + q_kw)

    cleaned: List[str] = []
    seen = set()
    for q in queries:
        q = compact_text(q)
        if len(q) < 4:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
        if len(cleaned) >= max_queries:
            break
    return cleaned


def build_ranking_query(record: Dict, claim: Dict) -> str:
    return compact_text(f"{record.get('question') or ''} {claim.get('sentence_text') or ''}")


def extract_key_terms_for_overlap(record: Dict, claim: Dict) -> List[str]:
    text = f"{record.get('question') or ''} {claim.get('sentence_text') or ''}"
    phrases = extract_quoted_phrases(text)
    terms = phrases[:]
    kws = extract_keywords(text, max_terms=14).split()
    terms.extend(kws)
    seen, out = set(), []
    for t in terms:
        key = t.lower()
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
