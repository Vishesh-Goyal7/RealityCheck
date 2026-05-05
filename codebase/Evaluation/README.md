# RealityCheck Phase 7 Evaluation

This module evaluates whether RealityCheck improves raw LLM answers using TruthfulQA ground truth.

It compares:

- `llm_original_answer`
- `realitycheck_corrected_answer`

against TruthfulQA `correct_answers` and `incorrect_answers`.

## Folder placement

Place this folder inside your `codebase/`:

```text
codebase/
  Phase_1/
  Phase_2/
  Phase_3/
  Phase_4/
  Phase_5/
  Phase_6/
  Phase_7_Evaluation/
```

## Expected inputs

1. Phase 6 output files, for example:

```text
../Phase_6/outputs/model_claims_llama_with_corrected_answers.json
../Phase_6/outputs/model_claims_granite_with_corrected_answers.json
```

2. TruthfulQA ground-truth file from Phase 2 or original dataset.

Supported formats:

- `.json`
- `.jsonl`
- `.csv`

The ground-truth file should contain:

- question ID if available: `id`, `question_id`, `qid`, `tqa_id`, or `sample_id`
- question text: `question` or `Question`
- correct answers: `correct_answers`, `Correct Answers`, `Best Answer`, etc.
- incorrect answers: `incorrect_answers`, `Incorrect Answers`, etc.

## Run

From `codebase/Phase_7_Evaluation`:

```bash
PYTHONPATH=src python src/evaluate_realitycheck.py \
  --phase6-inputs \
    ../Phase_6/outputs/model_claims_llama_with_corrected_answers.json \
    ../Phase_6/outputs/model_claims_granite_with_corrected_answers.json \
  --truthfulqa ../Phase_2/outputs/truthfulqa_eval_100.json \
  --output-dir outputs \
  --count-probable
```

If your TruthfulQA file is CSV:

```bash
PYTHONPATH=src python src/evaluate_realitycheck.py \
  --phase6-inputs ../Phase_6/outputs/model_claims_llama_with_corrected_answers.json \
  --truthfulqa ../Phase_2/data/TruthfulQA.csv \
  --output-dir outputs
```

## Optional semantic evaluation

If `sentence-transformers` is installed, you can use semantic similarity too:

```bash
PYTHONPATH=src python src/evaluate_realitycheck.py \
  --phase6-inputs ../Phase_6/outputs/model_claims_llama_with_corrected_answers.json \
  --truthfulqa ../Phase_2/outputs/truthfulqa_eval_100.json \
  --output-dir outputs \
  --use-semantic \
  --count-probable
```

If the model is not available, the script automatically falls back to lexical scoring.

## Outputs

The script creates:

```text
outputs/
  evaluation_summary.json
  evaluation_records.csv
  evaluation_report.md
```

## Metrics generated

### Answer-level metrics

- Raw LLM answer accuracy
- RealityCheck answer accuracy
- Absolute accuracy gain
- Relative accuracy gain
- Wrong-answer fix rate
- Overcorrection rate
- Correct-answer preservation rate
- Unresolved-after-correction count

### Behavior metrics

- Answer status distribution
- Raw label distribution
- RealityCheck label distribution
- Aggregate claim counts from Phase 6

## Important note

This evaluator does not call any LLM and does not use the internet. It is deterministic by default. This is intentional, because the evaluation should be reproducible for a research paper.
