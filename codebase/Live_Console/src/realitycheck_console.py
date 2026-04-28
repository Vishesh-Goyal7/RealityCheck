from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SUPPORTED_MODELS = {"llama", "granite", "mistral"}
DEFAULT_MODEL = "granite"


class PipelineError(RuntimeError):
    pass


def _now_run_id() -> str:
    return datetime.now().strftime("live_%Y%m%d_%H%M%S_%f")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _find_project_root(start: Optional[Path] = None) -> Path:
    """Find the codebase root containing Phase_2 ... Phase_6."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if all((candidate / f"Phase_{i}").exists() for i in range(2, 7)):
            return candidate
    raise PipelineError(
        "Could not find project root. Run this from inside your codebase folder, "
        "or pass --root /path/to/codebase."
    )


@contextmanager
def _temporary_sys_path(path: Path):
    path_str = str(path.resolve())
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _run_subprocess(cmd: List[str], *, cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print("\n$", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise PipelineError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _build_phase2_record(question: str, model_name: str, generation: Any) -> Dict[str, Any]:
    """Build the same shape Phase 3 already expects from Phase 2 outputs."""
    run_id = "live_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return {
        "id": run_id,
        "question": question,
        "category": "LiveUserQuestion",
        "question_type": "live_interactive",
        "source_dataset": "live_user_input",
        "model_name": generation.model_name,
        "model_id": generation.model_id,
        "system_prompt": generation.system_prompt,
        "prompt": generation.prompt,
        "raw_response": generation.raw_response,
        "cleaned_response": generation.cleaned_response,
        "response_quality_flags": generation.response_quality_flags,
        "response_quality": generation.response_quality,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "ground_truth_best_answer": None,
        "ground_truth_correct_answers": [],
        "ground_truth_incorrect_answers": [],
        "source": "",
        "mc1_choices": [],
        "mc1_labels": [],
        "mc2_choices": [],
        "mc2_labels": [],
        "raw_api_response": generation.raw_api_response,
    }


def generate_llm_answer(root: Path, question: str, model_name: str) -> Dict[str, Any]:
    """Use your existing Phase 2 WatsonX ResponseGenerator for one live question."""
    phase2_src = root / "Phase_2" / "src"
    if not phase2_src.exists():
        raise PipelineError(f"Phase 2 src folder not found: {phase2_src}")

    # Phase_2/src/config.py loads Phase_2/.env by itself.
    with _temporary_sys_path(phase2_src):
        from config import load_generation_config  # type: ignore
        from response_generator import ResponseGenerator  # type: ignore

        cfg = load_generation_config()
        generator = ResponseGenerator(cfg)
        generation = generator.generate(model_name=model_name, question=question)

    return _build_phase2_record(question, model_name, generation)


def _single_matching_json(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*{suffix}"))
    if not matches:
        raise PipelineError(f"No output file ending with {suffix!r} found in {folder}")
    if len(matches) > 1:
        # Usually this runner uses isolated run folders, so multiple matches means something odd.
        print(f"Warning: multiple matches in {folder}; using {matches[0].name}")
    return matches[0]


def run_pipeline_once(
    *,
    root: Path,
    question: str,
    model_name: str,
    run_dir: Path,
    top_pages: int = 3,
    candidate_pages: int = 8,
    top_chunks: int = 5,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Phase 2 -> 3 -> 4 -> 5 -> 6 for a single live question."""
    if model_name not in SUPPORTED_MODELS:
        raise PipelineError(f"Unsupported model {model_name!r}. Choose from: {sorted(SUPPORTED_MODELS)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== RealityCheck live run: {run_dir.name} | model={model_name} ===")

    # Phase 2: generate raw LLM answer using existing WatsonX client.
    print("\n[Phase 2] Generating LLM answer...")
    phase2_record = generate_llm_answer(root, question, model_name)
    phase2_out = run_dir / "phase2" / f"live_{model_name}.json"
    _save_json(phase2_out, [phase2_record])
    print(f"Saved Phase 2 live record -> {phase2_out}")

    # Phase 3: claim extraction.
    print("\n[Phase 3] Extracting claims...")
    phase3_out = run_dir / "phase3" / f"model_claims_{model_name}.json"
    _run_subprocess(
        [sys.executable, "src/phase3_runner.py", "--input", str(phase2_out), "--output", str(phase3_out)],
        cwd=root / "Phase_3",
        extra_env={"PYTHONPATH": "src"},
    )

    # Phase 4: blind retrieval. Do NOT pass --truthfulqa-data.
    print("\n[Phase 4] Retrieving evidence with blind/cache-first retrieval...")
    phase4_dir = run_dir / "phase4"
    _run_subprocess(
        [
            sys.executable,
            "src/phase4_runner.py",
            "--inputs",
            str(phase3_out),
            "--output-dir",
            str(phase4_dir),
            "--top-pages",
            str(top_pages),
            "--candidate-pages",
            str(candidate_pages),
            "--top-chunks",
            str(top_chunks),
        ],
        cwd=root / "Phase_4",
        extra_env={"PYTHONPATH": "src"},
    )
    phase4_out = _single_matching_json(phase4_dir, "_with_evidence_final.json")

    # Phase 5: verification.
    print("\n[Phase 5] Verifying claims...")
    phase5_dir = run_dir / "phase5"
    phase5_cmd = [
        sys.executable,
        "src/phase5_runner.py",
        "--inputs",
        str(phase4_out),
        "--output-dir",
        str(phase5_dir),
    ]
    if reference_date:
        phase5_cmd += ["--reference-date", reference_date]
    _run_subprocess(
        phase5_cmd,
        cwd=root / "Phase_5",
        extra_env={"PYTHONPATH": "src"},
    )
    phase5_out = _single_matching_json(phase5_dir, "_with_verification.json")

    # Phase 6: answer synthesis.
    print("\n[Phase 6] Synthesizing corrected answer...")
    phase6_dir = run_dir / "phase6"
    _run_subprocess(
        [
            sys.executable,
            "src/phase6_runner.py",
            "--inputs",
            str(phase5_out),
            "--output-dir",
            str(phase6_dir),
        ],
        cwd=root / "Phase_6",
        extra_env={"PYTHONPATH": "src"},
    )
    phase6_out = _single_matching_json(phase6_dir, "_with_corrected_answers.json")
    corrected_records = _load_json(phase6_out)
    if not corrected_records:
        raise PipelineError("Phase 6 produced no records.")

    result = corrected_records[0]
    result["_run_paths"] = {
        "run_dir": str(run_dir),
        "phase2": str(phase2_out),
        "phase3": str(phase3_out),
        "phase4": str(phase4_out),
        "phase5": str(phase5_out),
        "phase6": str(phase6_out),
    }
    return result


def print_final_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("REALITYCHECK RESULT")
    print("=" * 88)
    print(f"Question: {result.get('question', '')}")
    print(f"Model: {result.get('model_name', '')} ({result.get('model_id', '')})")
    print(f"Answer status: {result.get('answer_status', '')}")

    print("\nLLM original answer:")
    print((result.get("llm_original_answer") or "").strip() or "[empty]")

    print("\nRealityCheck corrected answer:")
    print((result.get("realitycheck_corrected_answer") or "").strip() or "[empty]")

    counts = result.get("claim_counts") or {}
    if counts:
        print("\nClaim counts:")
        print(
            f"supported={counts.get('supported', 0)}, "
            f"contradicted={counts.get('contradicted', 0)}, "
            f"insufficient={counts.get('insufficient_evidence', 0)}, "
            f"skipped={counts.get('skipped_non_checkable', 0)}, "
            f"total={counts.get('total', 0)}"
        )

    synthesis = result.get("claim_synthesis") or []
    if synthesis:
        print("\nClaim-level trace:")
        for item in synthesis:
            cid = item.get("claim_id", "claim")
            label = item.get("verification_label", "unknown")
            action = item.get("action", "unknown")
            print(f"- {cid}: {label} -> {action}")

    paths = result.get("_run_paths") or {}
    if paths:
        print("\nSaved outputs:")
        print(paths.get("run_dir", ""))
    print("=" * 88 + "\n")


def interactive_loop(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve() if args.root else _find_project_root()
    output_base = Path(args.output_base).resolve() if args.output_base else root / "Database" / "live_runs"
    model_name = args.model

    print("\nRealityCheck Interactive Console")
    print("Type a question and press Enter.")
    print("Commands: /exit, /quit, /model llama|granite|mistral, /help")
    print(f"Root: {root}")
    print(f"Live outputs: {output_base}")
    print(f"Current model: {model_name}\n")

    while True:
        try:
            user_input = input("RealityCheck > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            print("Exiting.")
            return
        if lowered == "/help":
            print("Commands: /exit, /quit, /model llama|granite|mistral")
            continue
        if lowered.startswith("/model"):
            parts = lowered.split()
            if len(parts) != 2 or parts[1] not in SUPPORTED_MODELS:
                print(f"Usage: /model {'|'.join(sorted(SUPPORTED_MODELS))}")
                continue
            model_name = parts[1]
            print(f"Current model changed to: {model_name}")
            continue

        run_id = _now_run_id()
        run_dir = output_base / run_id
        try:
            result = run_pipeline_once(
                root=root,
                question=user_input,
                model_name=model_name,
                run_dir=run_dir,
                top_pages=args.top_pages,
                candidate_pages=args.candidate_pages,
                top_chunks=args.top_chunks,
                reference_date=args.reference_date,
            )
            print_final_result(result)
        except Exception as exc:
            print("\nRealityCheck run failed:")
            print(str(exc))
            if args.debug:
                traceback.print_exc()
            print("Partial outputs, if any, are saved at:", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive terminal backend for the full RealityCheck pipeline.")
    parser.add_argument("--root", default="", help="Path to codebase root. Default: auto-detect from cwd.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--output-base", default="", help="Where to save live run artifacts. Default: Database/live_runs")
    parser.add_argument("--top-pages", type=int, default=3)
    parser.add_argument("--candidate-pages", type=int, default=8)
    parser.add_argument("--top-chunks", type=int, default=5)
    parser.add_argument("--reference-date", default="", help="Optional YYYY-MM-DD date for temporal verification.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    interactive_loop(args)


if __name__ == "__main__":
    main()
