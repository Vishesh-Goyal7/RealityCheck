# RealityCheck Phase 7 Evaluation Report

## Model: llama

- Records evaluated: 100
- Raw answer semantic accuracy: 0.550
- RealityCheck answer semantic accuracy: 0.220
- Average raw truth score: 0.602
- Average corrected truth score: 0.310
- Average truth score delta: -0.292
- Wrong-answer fix rate: 0.06666666666666667
- Overcorrection rate: 0.03636363636363636
- Safe abstentions: 56

### Outcome counts
- preserved_correct_answer: 20
- uncertain_change: 17
- safe_abstention: 56
- fixed_wrong_answer: 1
- improved_from_uncertain: 1
- still_wrong: 3
- overcorrected_good_answer: 2

## Model: granite

- Records evaluated: 100
- Raw answer semantic accuracy: 0.600
- RealityCheck answer semantic accuracy: 0.310
- Average raw truth score: 0.644
- Average corrected truth score: 0.389
- Average truth score delta: -0.255
- Wrong-answer fix rate: 0.1111111111111111
- Overcorrection rate: 0.016666666666666666
- Safe abstentions: 45

### Outcome counts
- fixed_wrong_answer: 2
- preserved_correct_answer: 29
- still_wrong: 7
- uncertain_change: 16
- safe_abstention: 45
- overcorrected_good_answer: 1
