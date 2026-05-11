import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _split_answers(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    # TruthfulQA CSV commonly separates answer variants by semicolon.
    if ';' in text:
        return [x.strip() for x in text.split(';') if x.strip()]
    # Some processed files may use pipes.
    if '|' in text:
        return [x.strip() for x in text.split('|') if x.strip()]
    return [text]


def load_truth_reference(path: str | Path) -> dict[str, dict]:
    """Load TruthfulQA reference from JSON/JSONL/CSV.

    Supported forms:
    1) JSON list with fields: id/question/correct_answers/incorrect_answers
    2) JSON dict containing records/data/items
    3) CSV with TruthfulQA columns such as Question, Best Answer, Correct Answers, Incorrect Answers

    Returns {id_or_question_key: {id, question, correct_answers, incorrect_answers}}
    """
    path = Path(path)
    suffix = path.suffix.lower()
    records = []

    if suffix in {'.json', '.jsonl'}:
        if suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f if line.strip()]
        else:
            obj = load_json(path)
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict):
                for key in ('records', 'data', 'items', 'questions'):
                    if isinstance(obj.get(key), list):
                        records = obj[key]
                        break
                if not records:
                    # Already keyed by id
                    records = list(obj.values())
    elif suffix == '.csv':
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError(f'Unsupported truth reference format: {path}')

    out = {}
    for idx, r in enumerate(records):
        rid = str(r.get('id') or r.get('question_id') or r.get('Question ID') or r.get('qid') or '').strip()
        question = str(r.get('question') or r.get('Question') or r.get('prompt') or '').strip()
        if not rid:
            # Many TruthfulQA exports do not have tqa ids; question key fallback is okay.
            rid = f'tqa_{idx:04d}' if not question else question.lower().strip()

        correct = []
        for k in ('correct_answers', 'correct_answer', 'Correct Answers', 'Correct Answer', 'Best Answer', 'best_answer', 'Best answer'):
            if k in r:
                correct.extend(_split_answers(r.get(k)))
        incorrect = []
        for k in ('incorrect_answers', 'incorrect_answer', 'Incorrect Answers', 'Incorrect Answer', 'Best Incorrect Answer', 'best_incorrect_answer'):
            if k in r:
                incorrect.extend(_split_answers(r.get(k)))

        # Deduplicate while preserving order.
        correct = list(dict.fromkeys([x for x in correct if x]))
        incorrect = list(dict.fromkeys([x for x in incorrect if x]))

        rec = {
            'id': str(r.get('id') or r.get('question_id') or rid),
            'question': question,
            'correct_answers': correct,
            'incorrect_answers': incorrect,
            'raw_reference': r,
        }
        out[str(rec['id'])] = rec
        if question:
            out[question.lower().strip()] = rec
    return out


def load_phase6_outputs(paths: list[str | Path]) -> list[dict]:
    all_records = []
    for path in paths:
        obj = load_json(path)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = obj.get('records') or obj.get('data') or obj.get('items') or []
            if not records and all(k in obj for k in ('llm_original_answer', 'realitycheck_corrected_answer')):
                records = [obj]
        else:
            raise ValueError(f'Unsupported Phase 6 output structure: {path}')
        for r in records:
            r = dict(r)
            r['_source_file'] = str(path)
            all_records.append(r)
    return all_records
