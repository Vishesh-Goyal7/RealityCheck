"""Run final RealityCheck Phase 4 evidence retrieval on one or more Phase 3 claim files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from embedder import EvidenceEmbedder
from retriever import EvidenceRetriever
from source_map import load_source_map
from wiki_client import SourceAwareClient


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def output_name_for(input_path: Path, output_dir: Path) -> Path:
    stem = input_path.stem
    stem = stem.replace("_with_evidence", "")
    return output_dir / f"{stem}_with_evidence_final.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="RealityCheck Phase 4 final source-aware evidence retrieval")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more Phase 3 claim JSON files")
    parser.add_argument("--output-dir", default="outputs", help="Directory for evidence outputs")
    parser.add_argument("--truthfulqa-data", default=None, help="Optional processed TruthfulQA JSON/CSV with id and source columns")
    parser.add_argument("--top-pages", type=int, default=3, help="Selected source/search pages per claim")
    parser.add_argument("--candidate-pages", type=int, default=8, help="Candidate pages considered before filtering")
    parser.add_argument("--top-chunks", type=int, default=5, help="Evidence chunks retained per claim")
    parser.add_argument("--sentences-per-chunk", type=int, default=3)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--embedding-model", default="sentence-transformers/multi-qa-MiniLM-L6-dot-v1")
    parser.add_argument("--min-chunk-similarity", type=float, default=0.50)
    parser.add_argument("--min-page-similarity", type=float, default=0.15)
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    source_map = load_source_map(args.truthfulqa_data)

    source_client = SourceAwareClient()
    embedder = EvidenceEmbedder(model_name=args.embedding_model)
    retriever = EvidenceRetriever(
        source_client=source_client,
        embedder=embedder,
        top_pages=args.top_pages,
        candidate_pages=args.candidate_pages,
        top_chunks=args.top_chunks,
        sentences_per_chunk=args.sentences_per_chunk,
        overlap_sentences=args.overlap_sentences,
        min_chunk_similarity=args.min_chunk_similarity,
        min_page_similarity=args.min_page_similarity,
    )

    summary = []
    for input_file in args.inputs:
        input_path = Path(input_file)
        records = load_json(input_path)
        if args.limit_records is not None:
            records = records[: args.limit_records]

        enriched_records = []
        print(f"\nProcessing {input_path} ({len(records)} records)")
        for record in tqdm(records, desc=input_path.name):
            source_url = record.get("source") or source_map.get(str(record.get("id")), "")
            enriched_records.append(retriever.retrieve_for_record(record, source_url=source_url))

        out_json = output_name_for(input_path, output_dir)
        save_json(out_json, enriched_records)

        evs = [ev for r in enriched_records for ev in r.get("evidence_results", [])]
        status_counts: Dict[str, int] = {}
        flag_counts: Dict[str, int] = {}
        for ev in evs:
            status = ev.get("retrieval_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            for flag in ev.get("retrieval_quality_flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        summary.append(
            {
                "input_file": str(input_path),
                "output_json": str(out_json),
                "records": len(enriched_records),
                "total_claims": len(evs),
                "status_counts": status_counts,
                "flag_counts": flag_counts,
                "claims_with_strong_evidence": status_counts.get("ok", 0),
                "claims_with_weak_evidence": status_counts.get("weak_evidence", 0),
                "claims_without_evidence": status_counts.get("no_evidence_found", 0),
                "skipped_non_checkable": status_counts.get("skipped_non_checkable", 0),
            }
        )
        print(f"Saved: {out_json}")

    save_json(output_dir / "phase4_summary_final.json", summary)
    print("\nPhase 4 final retrieval complete. Summary saved to", output_dir / "phase4_summary_final.json")


if __name__ == "__main__":
    main()
