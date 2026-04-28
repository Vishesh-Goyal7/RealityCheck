# RealityCheck Phase 5 Multi-Fact Patch

This patch upgrades Phase 5 for the cache-first/blind Phase 4 outputs.

## Main fix
Phase 5 now uses `evidence_query` as the internal reasoning hypothesis for short answer fragments such as:

- `Euros`
- `Euros and kroner`
- `Baseball`

The public output still preserves the original `sentence_text`, but the verifier no longer loses the original question context.

## Added logic
- Multi-fact support aggregation for compound answers.
  - Example: Germany + Norway currency answer can be supported by separate Germany/euro and Norway/krone chunks.
- Guard against false contradictions from unrelated negated entities.
  - Example: Edson patent claim should not be contradicted merely because another chunk says Carver did not invent peanut butter.
- Evidence gap analysis now uses the reasoning text, so repair queries are less stupid for short answers.

## Expected correction
The previous false case:

```text
Question: On a visit to Germany and Norway, you'll need to change dollars into what?
Answer: Euros and kroner.
Old label: insufficient_evidence
New label: supported
```

should now pass through `compound_fact_support_override`.
