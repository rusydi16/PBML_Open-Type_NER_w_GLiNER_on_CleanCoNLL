# Findings: Fair Evaluation of GLiNER on CleanCoNLL vs CoNLL-03

## 1. Performance Comparison

Overall F1 on CoNLL-03: **61.26%** | Overall F1 on CleanCoNLL: **58.71%** | Delta: **-2.55** pp (lower on CleanCoNLL).

### Per-Entity-Type Breakdown

- **PER**: CoNLL-03 F1 = 80.92%, CleanCoNLL F1 = 81.88%, Delta = 0.96 pp (higher)
- **ORG**: CoNLL-03 F1 = 47.74%, CleanCoNLL F1 = 46.40%, Delta = -1.34 pp (lower)
- **LOC**: CoNLL-03 F1 = 65.80%, CleanCoNLL F1 = 61.57%, Delta = -4.23 pp (lower)
- **MISC**: CoNLL-03 F1 = 0.00%, CleanCoNLL F1 = 0.00%, Delta = 0.00 pp (unchanged)

## 2. Error Category Changes

| Error Category | CoNLL-03 | CleanCoNLL | Delta |
|---|---|---|---|
| type_error | 858 | 1069 | +211 |
| boundary_error | 215 | 183 | -32 |
| type_boundary_error | 337 | 351 | +14 |
| missing | 691 | 698 | +7 |
| spurious | 976 | 913 | -63 |

## 3. Noise Attribution

- **correct_both**: 3224 — Predictions correct under both annotation sets
- **noise_penalized_correct**: 107 — Correct predictions penalised due to noisy CoNLL-03 labels
- **model_learned_noise**: 236 — Model learned noisy patterns from training data
- **genuine_error**: 2232 — Genuine model errors (wrong under both annotations)
- **missed_both**: 1799 — Entities missed under both annotation sets
- **missed_conll_only**: 220 — Entities missed only when evaluating against CoNLL-03
- **missed_clean_only**: 430 — Entities missed only when evaluating against CleanCoNLL

## 4. Conclusion

When evaluated against CleanCoNLL, GLiNER's overall F1 is 2.55 percentage points lower than on the original CoNLL-03 test set. This indicates that some predictions the model gets 'right' on CoNLL-03 may actually be matching noisy labels. Noise attribution analysis found 236 cases where the model appears to have learned noisy patterns.
