"""Date/age verification utilities for RealityCheck Phase 5.

This module handles a narrow but important class of TruthfulQA/indexical claims:
claims of the form "X is N years old" with evidence containing a birth date.
It avoids sending obvious age claims to manual review when deterministic arithmetic
is enough.

Patch notes:
- Do NOT simply use the first/highest-scoring date in a chunk. Biography pages often
  contain dates for parents, marriages, elections, deaths, etc.
- Prefer dates explicitly present in the claim, then dates appearing in birth contexts
  such as "X was born on ..." or "(born Month Day, Year)".
- Penalize dates near parent/father/mother/marriage/death contexts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional, Dict, Any, Iterable, List, Tuple

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DATE_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"(?P<day>\d{1,2}),\s*(?P<year>\d{4})\b",
    flags=re.I,
)

AGE_RE = re.compile(r"\b(?P<age>\d{1,3})\s+years?\s+old\b", flags=re.I)
AS_OF_YEAR_RE = re.compile(r"\bas\s+of\s+(?P<year>\d{4})\b", flags=re.I)


@dataclass
class TemporalDecision:
    attempted: bool
    label: Optional[str]
    reason: str
    claimed_age: Optional[int] = None
    computed_age: Optional[int] = None
    birth_date: Optional[str] = None
    reference_date: Optional[str] = None
    evidence_chunk_id: Optional[str] = None
    evidence_page_title: Optional[str] = None
    date_selection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_reference_date(value: str | None = None) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return date.today()


def _date_from_match(match: re.Match) -> Optional[date]:
    mon = MONTHS[match.group("month").lower()]
    day = int(match.group("day"))
    year = int(match.group("year"))
    try:
        return date(year, mon, day)
    except ValueError:
        return None


def _parse_birth_date(text: str) -> Optional[date]:
    match = DATE_RE.search(text or "")
    if not match:
        return None
    return _date_from_match(match)


def _all_dates_with_context(text: str) -> List[Tuple[date, str, int]]:
    out: List[Tuple[date, str, int]] = []
    raw = text or ""
    for match in DATE_RE.finditer(raw):
        d = _date_from_match(match)
        if not d:
            continue
        start = max(0, match.start() - 90)
        end = min(len(raw), match.end() + 90)
        out.append((d, raw[start:end], match.start()))
    return out


def _compute_age(born: date, ref: date) -> int:
    age = ref.year - born.year
    if (ref.month, ref.day) < (born.month, born.day):
        age -= 1
    return age


def _reference_date_for_claim(claim: str, default_reference_date: date) -> date:
    """Use Dec 31 for vague 'as of YYYY' claims.

    'As of 2022' is underspecified; evaluating at year-end is the least surprising
    deterministic convention for dataset-style statements.
    """
    m = AS_OF_YEAR_RE.search(claim or "")
    if m:
        return date(int(m.group("year")), 12, 31)
    return default_reference_date


def _claimed_birth_dates(claim: str) -> List[date]:
    return [d for d, _ctx, _pos in _all_dates_with_context(claim or "")]


def _score_date_candidate(
    claim: str,
    cv: Dict[str, Any],
    candidate_date: date,
    context: str,
    claimed_dates: List[date],
) -> Tuple[float, str]:
    text = str(cv.get("text", ""))
    page_title = str(cv.get("page_title", ""))
    score = float(cv.get("phase5_usable_score", 0.0) or 0.0)
    ctx = (context or "").lower()
    claim_l = (claim or "").lower()
    page_l = page_title.lower()

    reasons: List[str] = []

    # Most reliable: the date appears in the claim and evidence.
    if candidate_date in claimed_dates:
        score += 1.25
        reasons.append("date_matches_claim")

    # Good birth-date contexts.
    if re.search(r"\b(was|is)?\s*born\s+(on\s+)?$", ctx[: max(0, ctx.find(str(candidate_date.year)))]) or "born" in ctx:
        score += 0.55
        reasons.append("birth_context")
    if "(born" in ctx:
        score += 0.45
        reasons.append("parenthetical_born_context")

    # Entity/page support. For claims like Barack Obama, the main biography page is reliable.
    claim_name_tokens = set(re.findall(r"\b[A-Z][a-z]+\b", claim or ""))
    if claim_name_tokens and all(tok.lower() in page_l for tok in list(claim_name_tokens)[:2]):
        score += 0.20
        reasons.append("entity_page_match")

    # Penalize unrelated date contexts.
    bad_context_terms = [
        "parents met", "father", "mother", "married", "death", "died", "killed",
        "inauguration", "presidency", "election", "approval", "visited", "library",
    ]
    for term in bad_context_terms:
        if term in ctx:
            score -= 0.45
            reasons.append(f"penalized_{term.replace(' ', '_')}")

    # If claim says born on a date and candidate does not match it, be harsh.
    if claimed_dates and candidate_date not in claimed_dates:
        score -= 1.30
        reasons.append("candidate_date_conflicts_with_claim_date")

    # If text contains a direct compact infobox-style statement, boost.
    if re.search(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+\(born\s+", text):
        score += 0.50
        reasons.append("infobox_style_born_statement")

    return score, "+".join(reasons) if reasons else "generic_date_candidate"


def verify_age_claim(
    claim: str,
    chunk_verifications: Iterable[Dict[str, Any]],
    *,
    reference_date: date,
) -> TemporalDecision:
    age_match = AGE_RE.search(claim or "")
    if not age_match:
        return TemporalDecision(False, None, "not_an_explicit_age_claim")

    claimed_age = int(age_match.group("age"))
    ref_date = _reference_date_for_claim(claim, reference_date)
    claimed_dates = _claimed_birth_dates(claim)

    best = None
    best_score = -999.0
    best_reason = ""

    for cv in chunk_verifications:
        text = str(cv.get("text", ""))
        for born, ctx, _pos in _all_dates_with_context(text):
            # Avoid using weak unrelated chunks unless exact date appears in claim.
            candidate_score, reason = _score_date_candidate(claim, cv, born, ctx, claimed_dates)
            if candidate_score > best_score:
                best_score = candidate_score
                best = (cv, born)
                best_reason = reason

    if not best or best_score < 0.70:
        return TemporalDecision(
            True,
            "needs_manual_review",
            "age_claim_detected_but_no_reliable_birth_date_evidence",
            claimed_age=claimed_age,
            reference_date=ref_date.isoformat(),
            date_selection_reason=best_reason or "no_candidate_birth_date_found",
        )

    cv, born = best
    computed = _compute_age(born, ref_date)
    label = "supported" if computed == claimed_age else "contradicted"
    reason = (
        f"Age claim verified using retrieved birth date evidence. "
        f"Computed age is {computed} on {ref_date.isoformat()}, claimed age is {claimed_age}."
    )
    return TemporalDecision(
        True,
        label,
        reason,
        claimed_age=claimed_age,
        computed_age=computed,
        birth_date=born.isoformat(),
        reference_date=ref_date.isoformat(),
        evidence_chunk_id=cv.get("chunk_id"),
        evidence_page_title=cv.get("page_title"),
        date_selection_reason=best_reason,
    )
