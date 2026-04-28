"""Preflight validation for RealityCheck Phase 5.

This module catches the exact failure mode we hit: Phase 5 being run on a stale,
partial, or accidentally regenerated Phase 4 file where evidence chunks vanished.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple


def evidence_coverage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    checkable = 0
    skipped = 0
    with_chunks = 0
    ok_status = 0
    no_evidence = 0
    status_counts: Counter[str] = Counter()

    suspicious_examples: List[Dict[str, Any]] = []

    for record in records:
        for er in record.get("evidence_results", []) or []:
            total += 1
            status = str(er.get("retrieval_status", "missing_status"))
            status_counts[status] += 1

            if not er.get("is_checkable", True) or status == "skipped_non_checkable":
                skipped += 1
                continue

            checkable += 1
            chunks = er.get("evidence_chunks", []) or []
            if status == "ok":
                ok_status += 1
            if chunks:
                with_chunks += 1
            if status == "no_evidence_found" or not chunks:
                no_evidence += 1
                if len(suspicious_examples) < 5:
                    suspicious_examples.append(
                        {
                            "id": record.get("id"),
                            "claim_id": er.get("claim_id"),
                            "question": record.get("question"),
                            "source_url_used": er.get("source_url_used"),
                            "retrieval_status": status,
                            "chunk_count": len(chunks),
                        }
                    )

    chunk_coverage = (with_chunks / checkable) if checkable else 1.0
    ok_ratio = (ok_status / checkable) if checkable else 1.0

    return {
        "total_evidence_items": total,
        "checkable_items": checkable,
        "skipped_non_checkable": skipped,
        "items_with_chunks": with_chunks,
        "ok_status_items": ok_status,
        "items_without_evidence": no_evidence,
        "chunk_coverage_ratio": chunk_coverage,
        "ok_status_ratio": ok_ratio,
        "status_counts": dict(status_counts),
        "suspicious_examples": suspicious_examples,
    }


def validate_phase4_input(
    records: List[Dict[str, Any]],
    *,
    min_chunk_coverage: float = 0.50,
    strict: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """Validate that Phase 4 evidence is actually present before NLI.

    Returns: (is_valid, stats, message)
    """
    stats = evidence_coverage(records)

    checkable = stats["checkable_items"]
    if checkable == 0:
        return True, stats, "No checkable claims found; Phase 5 can safely skip verification."

    coverage = stats["chunk_coverage_ratio"]
    if coverage < min_chunk_coverage:
        msg = (
            f"Phase 4 evidence coverage is too low for Phase 5: "
            f"{stats['items_with_chunks']}/{checkable} checkable claims contain evidence chunks "
            f"({coverage:.1%}), below required {min_chunk_coverage:.1%}. "
            "This usually means you passed a stale/wrong Phase 4 file or reran Phase 4 while Wikipedia/external fetch failed. "
            "Re-run Phase 4 final and verify phase4_summary_final.json before running Phase 5."
        )
        return (not strict), stats, msg

    return True, stats, "Phase 4 evidence coverage looks acceptable."
