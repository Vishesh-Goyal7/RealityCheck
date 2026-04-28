"""RealityCheck Phase 5 runner.

Reads Phase 4 evidence-enriched JSON files and writes verification-enriched outputs.
This calibrated version validates evidence coverage before running NLI, so stale or
broken Phase 4 files do not silently produce useless `insufficient_evidence` outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from claim_verifier import ClaimVerifier
from io_utils import load_json, save_json
from preflight import validate_phase4_input


def derive_output_name(input_path: str | Path) -> str:
    stem = Path(input_path).stem
    if stem.endswith("_with_evidence_final"):
        stem = stem.replace("_with_evidence_final", "")
    elif stem.endswith("_with_evidence"):
        stem = stem.replace("_with_evidence", "")
    return f"{stem}_with_verification"


def summarize_file(
    input_file: str,
    output_json: str,
    records: List[Dict[str, Any]],
    preflight_stats: Dict[str, Any],
) -> Dict[str, Any]:
    label_counts = Counter()
    evidence_strength_counts = Counter()
    total_claims = 0

    for record in records:
        for result in record.get("verification_results", []) or []:
            total_claims += 1
            label_counts[result.get("verification_label", "unknown")] += 1
            evidence_strength_counts[result.get("evidence_strength", "unknown")] += 1

    return {
        "input_file": input_file,
        "output_json": output_json,
        "records": len(records),
        "total_verified_items": total_claims,
        "label_counts": dict(label_counts),
        "evidence_strength_counts": dict(evidence_strength_counts),
        "phase4_preflight": preflight_stats,
    }


def run_for_file(
    input_path: str,
    output_dir: Path,
    verifier: ClaimVerifier,
    *,
    limit_records: int | None = None,
    min_input_evidence_coverage: float = 0.50,
    no_strict_preflight: bool = False,
) -> Dict[str, Any]:
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {input_path}")

    if limit_records is not None:
        data = data[:limit_records]

    valid, stats, message = validate_phase4_input(
        data,
        min_chunk_coverage=min_input_evidence_coverage,
        strict=not no_strict_preflight,
    )

    print(f"\nPreflight for {input_path}:")
    print(message)
    print(stats)

    if not valid:
        raise RuntimeError(
            message
            + "\n\nUse the final Phase 4 outputs, e.g. "
            + "model_claims_llama_with_evidence_final.json and model_claims_mistral_with_evidence_final.json. "
            + "Do not pass Phase 3 files, old Phase 4 files, or Phase 5 verification files as input."
        )

    verified_records: List[Dict[str, Any]] = []
    print(f"\nProcessing {input_path} ({len(data)} records)")

    for record in tqdm(data, desc=Path(input_path).name):
        verified_records.append(verifier.verify_record(record))

    output_name = derive_output_name(input_path)
    output_json = output_dir / f"{output_name}.json"

    save_json(output_json, verified_records)

    return summarize_file(str(input_path), str(output_json), verified_records, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="RealityCheck Phase 5 claim verification runner")
    parser.add_argument("--inputs", nargs="+", required=True, help="Phase 4 evidence JSON files")
    parser.add_argument("--output-dir", default="outputs", help="Directory for Phase 5 outputs")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--support-threshold", type=float, default=0.65)
    parser.add_argument("--contradiction-threshold", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--reference-date", default=None, help="Reference date for temporal/age verification, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--min-evidence-score", type=float, default=0.52)
    parser.add_argument("--min-input-evidence-coverage", type=float, default=0.50)
    parser.add_argument("--review-margin", type=float, default=0.20, help="If max NLI score crosses this but fails safety gates, mark needs_manual_review")
    parser.add_argument("--no-strict-preflight", action="store_true", help="Warn instead of failing when Phase 4 evidence coverage is suspiciously low")
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NLI verifier: {args.nli_model}")
    verifier = ClaimVerifier(
        nli_model_name=args.nli_model,
        support_threshold=args.support_threshold,
        contradiction_threshold=args.contradiction_threshold,
        margin=args.margin,
        max_chunks=args.max_chunks,
        min_evidence_score=args.min_evidence_score,
        review_margin=args.review_margin,
        reference_date=args.reference_date,
    )

    summaries: List[Dict[str, Any]] = []
    for input_path in args.inputs:
        summaries.append(
            run_for_file(
                input_path,
                output_dir,
                verifier,
                limit_records=args.limit_records,
                min_input_evidence_coverage=args.min_input_evidence_coverage,
                no_strict_preflight=args.no_strict_preflight,
            )
        )

    summary_path = output_dir / "phase5_summary.json"
    save_json(summary_path, summaries)

    print("\nPhase 5 complete.")
    print(f"Summary saved to: {summary_path}")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
