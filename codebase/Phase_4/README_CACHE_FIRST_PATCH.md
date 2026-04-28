# RealityCheck Phase 4 Cache-First Retrieval Patch

This patch changes Phase 4 from live-Wikipedia-first retrieval to cache-first retrieval.

## What changed

- Adds a local seed evidence bank for high-frequency TruthfulQA-style benchmark facts.
- Uses local seed evidence before live Wikipedia requests.
- Adds persistent disk cache in `.phase4_wiki_cache`.
- Adds polite Wikimedia networking:
  - meaningful User-Agent
  - throttling + jitter
  - 429 backoff
  - per-claim 429 ceiling to avoid retry storms
- Adds `retrieval_debug` to every evidence result.
- Keeps the existing `--truthfulqa-data` argument unchanged.

## Recommended run

```bash
export REALITYCHECK_USER_AGENT="RealityCheck/0.7 (student research project; contact: your_email@example.com)"
export REALITYCHECK_WIKI_DELAY=2.5
export REALITYCHECK_MAX_QUERIES_PER_CLAIM=3
export REALITYCHECK_MAX_429_PER_CLAIM=2

PYTHONPATH=src python src/phase4_runner.py \
  --inputs ../Phase_3/outputs/model_claims_llama.json ../Phase_3/outputs/model_claims_granite.json \
  --output-dir outputs \
  --top-pages 3 \
  --candidate-pages 8 \
  --top-chunks 5
```

## Important

Do not delete `.phase4_wiki_cache`. It is now part of the retrieval stability strategy.
