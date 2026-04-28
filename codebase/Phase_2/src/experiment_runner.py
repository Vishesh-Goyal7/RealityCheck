from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import load_generation_config, load_paths_config
from dataset_loader import TruthfulQALoader
from response_generator import ResponseGenerator


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_record(item: dict[str, Any], generation: Any) -> dict[str, Any]:
    return {
        'id': item['id'],
        'question': item['question'],
        'category': item.get('category'),
        'question_type': item.get('question_type'),
        'source_dataset': item.get('source_dataset', 'TruthfulQA'),
        'model_name': generation.model_name,
        'model_id': generation.model_id,
        'system_prompt': generation.system_prompt,
        'prompt': generation.prompt,
        'raw_response': generation.raw_response,
        'cleaned_response': generation.cleaned_response,
        'response_quality_flags': generation.response_quality_flags,
        'response_quality': generation.response_quality,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ground_truth_best_answer': item.get('best_answer'),
        'ground_truth_correct_answers': item.get('correct_answers', []),
        'ground_truth_incorrect_answers': item.get('incorrect_answers', []),
        'source': item.get('source'),
        'mc1_choices': item.get('mc1_choices', []),
        'mc1_labels': item.get('mc1_labels', []),
        'mc2_choices': item.get('mc2_choices', []),
        'mc2_labels': item.get('mc2_labels', []),
        'raw_api_response': generation.raw_api_response,
    }


def save_json(path: Path, payload: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run RealityCheck Phase 2 generation experiments.')
    parser.add_argument("--model", type=str, required=True, choices=["llama", "mistral", "granite"])
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-path', type=str, default='')
    parser.add_argument('--output-prefix', type=str, default='')
    args = parser.parse_args()

    cfg = load_generation_config()
    paths = load_paths_config()
    ensure_dir(paths.data_dir)
    ensure_dir(paths.outputs_dir)

    dataset_path = Path(args.dataset_path) if args.dataset_path else paths.processed_dataset_path
    loader = TruthfulQALoader(dataset_path)
    data = loader.sample(sample_size=args.sample_size, seed=args.seed)
    generator = ResponseGenerator(cfg)

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        try:
            generation = generator.generate(model_name=args.model, question=item['question'])
            records.append(build_output_record(item, generation))
            print(f'[{idx}/{len(data)}] OK -> {item["id"]} ({generation.response_quality})')
        except Exception as exc:
            print(f'[{idx}/{len(data)}] FAILED -> {item.get("id", "unknown")}: {exc}')

    prefix = args.output_prefix or f'{args.model}_{len(records)}_chatfixed'
    save_json(paths.outputs_dir / f'{prefix}.json', records)
    print(f'Saved {len(records)} records to {paths.outputs_dir}')


if __name__ == '__main__':
    main()
