from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from answer_synthesizer import synthesize_file
from io_utils import load_json, save_json


def _output_name(input_path: Path) -> str:
    name = input_path.name
    if name.endswith("_with_verification.json"):
        return name.replace("_with_verification.json", "_with_corrected_answers.json")
    if name.endswith(".json"):
        return name[:-5] + "_with_corrected_answers.json"
    return name + "_with_corrected_answers.json"


def _summarize(output_records: List[Dict[str, Any]], input_file: str, output_file: str) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    total_claim_counts = {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
        "skipped_non_checkable": 0,
        "total": 0,
    }
    changed = 0
    for rec in output_records:
        status = rec.get("answer_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if (rec.get("llm_original_answer") or "").strip() != (rec.get("realitycheck_corrected_answer") or "").strip():
            changed += 1
        for k, v in (rec.get("claim_counts") or {}).items():
            if k in total_claim_counts:
                total_claim_counts[k] += int(v or 0)
    return {
        "input_file": input_file,
        "output_json": output_file,
        "records": len(output_records),
        "answers_changed": changed,
        "answer_status_counts": status_counts,
        "aggregate_claim_counts": total_claim_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: synthesize corrected answers from Phase 5 verification outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Phase 5 verification JSON files")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save Phase 6 outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for input_file in args.inputs:
        in_path = Path(input_file)
        data = load_json(in_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list of records in {input_file}")
        synthesized = synthesize_file(data)
        out_path = output_dir / _output_name(in_path)
        save_json(synthesized, out_path)
        summaries.append(_summarize(synthesized, input_file, str(out_path)))
        print(f"Saved {len(synthesized)} corrected-answer records -> {out_path}")

    save_json(summaries, output_dir / "phase6_summary.json")
    print(f"Saved Phase 6 summary -> {output_dir / 'phase6_summary.json'}")


if __name__ == "__main__":
    main()
