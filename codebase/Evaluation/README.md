# RealityCheck Phase 7 — Semantic Evaluation

This phase evaluates whether RealityCheck improves answer truthfulness over the raw LLM output.

It compares each answer against TruthfulQA reference answers using semantic similarity, not just word overlap.

## Inputs

1. Phase 6 corrected answer files, for example:
   - `../Phase_6/outputs/model_claims_llama_with_corrected_answers.json`
   - `../Phase_6/outputs/model_claims_granite_with_corrected_answers.json`

2. A TruthfulQA reference file from Phase 2 containing:
   - `id` or `question`
   - `correct_answers`
   - `incorrect_answers`

CSV TruthfulQA exports are also supported if they contain columns like:
- `Question`
- `Best Answer`
- `Correct Answers`
- `Incorrect Answers`

## Install

From your main `codebase` folder:

```bash
source venv/bin/activate
pip install sentence-transformers scikit-learn numpy pandas
```

If embeddings fail, the script automatically falls back to lexical similarity. You can also force fallback:

```bash
--disable-embeddings
```

## Run

```bash
PYTHONPATH=src python src/phase7_runner.py \
  --phase6-inputs ../Phase_6/outputs/model_claims_llama_with_corrected_answers.json ../Phase_6/outputs/model_claims_granite_with_corrected_answers.json \
  --truth-reference ../Phase_2/outputs/truthfulqa_processed.json \
  --output-dir outputs
```

Adjust `--truth-reference` to whatever your Phase 2 processed TruthfulQA reference file is called.

## Outputs

- `outputs/phase7_summary.json`
- `outputs/phase7_detailed_results.json`
- `outputs/phase7_detailed_results.csv`
- `outputs/phase7_report.md`

## Main Metrics

For each model:

- Raw answer semantic accuracy
- RealityCheck answer semantic accuracy
- Average raw truth score
- Average corrected truth score
- Average truth score delta
- Wrong-answer fix rate
- Overcorrection rate
- Safe abstention count
- Outcome counts

## How scoring works

For each answer:

1. Compute semantic similarity to TruthfulQA correct answers.
2. Compute semantic similarity to TruthfulQA incorrect answers.
3. Use the margin:

```text
truth_margin = similarity_to_correct - similarity_to_incorrect
```

A corrected answer is judged better when it is closer to correct references and farther from incorrect references.

This is not perfect, but it is far better than exact word matching.
