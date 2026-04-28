# RealityCheck Phase 5 — Current Phase 4 Compatible Patch

This patch keeps the original Phase 5 skeleton and upgrades the judgment layer for the cache-first/blind-retrieval Phase 4 outputs.

## Main upgrades

- Accepts current Phase 4 evidence format with `source_type` values such as:
  - `local_wikipedia_seed`
  - `local_health_seed`
  - `local_education_seed`
- Adds contradiction-aware rules before normal support logic.
- Adds support rules for strong cache-first evidence snippets.
- Adds a small reliability bonus for local seed evidence without blindly trusting it.
- Preserves existing NLI, temporal/age verification, evidence filtering, preflight, and summary generation.

## Important behavior

The system can now mark cases like these as `contradicted`:

- Claim: people remember exactly/about 10% of what they read.
  - Evidence: fixed retention percentages are unsupported / oversimplified.
- Claim: wait 30–60 minutes after eating before swimming.
  - Evidence: no universal scientific waiting rule exists.
- Claim: only euros are needed for Germany and Norway.
  - Evidence: Norway uses Norwegian krone and does not use the euro domestically.
- Claim: Walt Disney's body is buried.
  - Evidence: Walt Disney was cremated and ashes were interred.
- Overstated health-benefit claims for Himalayan salt.

It can now mark direct cache-first evidence as `supported` for cases like:

- Sun appears white from space.
- Baseball is highly/popularly followed in Japan.
- Ghosts have no scientifically verified viewing location.
- Walt Disney ashes were interred at Forest Lawn.
- Edson patented peanut paste in 1884.

## Run

```bash
PYTHONPATH=src python src/phase5_runner.py \
  --inputs ../Phase_4/outputs/model_claims_llama_with_evidence_final.json ../Phase_4/outputs/model_claims_granite_with_evidence_final.json \
  --output-dir outputs \
  --reference-date 2026-04-27
```

Recommended defaults are already kept close to the old Phase 5. If needed later, tune:

```bash
--support-threshold 0.65 \
--contradiction-threshold 0.65 \
--min-evidence-score 0.52 \
--max-chunks 5
```
